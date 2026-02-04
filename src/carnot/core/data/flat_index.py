"""
Flat file index for LLM-based query routing.

This module provides a simple flat index where all file summaries are provided
to an LLM to select relevant files for a given query.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import litellm
import tiktoken

from carnot.core.data.smv import FileSummary

logger = logging.getLogger(__name__)


class FlatFileIndex:
    """
    Flat file routing index that uses LLM to select relevant files.
    All file summaries are provided to the LLM in a single context.
    """
    
    def __init__(self, file_summaries: List[FileSummary], context_limit_tokens: int = 120000):
        """
        Initialize the flat file index.
        
        Args:
            file_summaries: List of FileSummary objects for all files
            context_limit_tokens: Maximum tokens to use for LLM context (with safety margin)
        """
        self.file_summaries = {fs.file_id: fs for fs in file_summaries}
        self.context_limit = context_limit_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Choose model based on available API keys
        self.model_id = "openai/gpt-4o-mini"
        if os.getenv("ANTHROPIC_API_KEY"):
            self.model_id = "anthropic/claude-3-5-sonnet-20241022"
        elif os.getenv("GEMINI_API_KEY"):
            self.model_id = "vertex_ai/gemini-2.0-flash-001"
        elif os.getenv("TOGETHER_API_KEY"):
            self.model_id = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
        
        logger.info(f"Initialized FlatFileIndex with {len(self.file_summaries)} files using model {self.model_id}")
    
    def route(
        self, 
        query: str, 
        top_k: int = 10,
        max_files: Optional[int] = None,
        use_embeddings_fallback: bool = True
    ) -> List[str]:
        """
        Route a query to the most relevant files using LLM.
        
        Args:
            query: The search query
            top_k: Number of files to return
            max_files: Maximum number of files to consider (for context limit)
            use_embeddings_fallback: If True, fallback to embedding similarity if context too large
        
        Returns:
            List of file IDs for the most relevant files
        """
        logger.info(f"=== Starting routing for query: '{query}' ===")
        logger.info(f"Requesting top_k={top_k} files from {len(self.file_summaries)} total files")
        
        if not self.file_summaries:
            logger.warning("No file summaries available for routing")
            return []
        
        # Limit files if specified
        summaries_to_use = list(self.file_summaries.values())
        if max_files and len(summaries_to_use) > max_files:
            logger.info(f"Limiting routing to {max_files} files (total: {len(summaries_to_use)})")
            summaries_to_use = summaries_to_use[:max_files]
        
        # Build prompt
        logger.info(f"Building prompt with {len(summaries_to_use)} file summaries...")
        prompt = self._format_summaries_for_llm(query, summaries_to_use, top_k)
        
        # Check if prompt fits in context
        prompt_tokens = len(self.tokenizer.encode(prompt))
        logger.info(f"Prompt size: {prompt_tokens} tokens (limit: {self.context_limit})")
        
        if prompt_tokens > self.context_limit:
            logger.warning(
                f"Prompt ({prompt_tokens} tokens) exceeds context limit ({self.context_limit}). "
                f"Consider reducing max_files or using hierarchical index."
            )
            if use_embeddings_fallback:
                logger.info("Falling back to embedding similarity routing")
                return self._embedding_fallback_route(query, summaries_to_use, top_k)
            else:
                # Just return first top_k files
                return [fs.file_id for fs in summaries_to_use[:top_k]]
        
        logger.info(f"Calling LLM for routing with model: {self.model_id}")
        
        # Call LLM for routing
        try:
            routed_ids = self._llm_route(prompt, top_k, summaries_to_use)
            logger.info(f"=== Routing complete: {len(routed_ids)} files selected ===")
            if routed_ids:
                logger.info(f"First 5 routed files: {routed_ids[:5]}")
            return routed_ids
        except Exception as e:
            logger.error(f"LLM routing failed with exception: {e}")
            if use_embeddings_fallback:
                logger.info("Falling back to embedding similarity routing")
                return self._embedding_fallback_route(query, summaries_to_use, top_k)
            else:
                return []
    
    def _format_summaries_for_llm(self, query: str, summaries: List[FileSummary], top_k: int) -> str:
        """
        Format file summaries into an LLM prompt for routing.
        
        Args:
            query: The search query
            summaries: List of FileSummary objects
            top_k: Number of files to select
        
        Returns:
            Formatted prompt string
        """
        # Build numbered list of summaries with clean file paths
        summary_lines = []
        for i, fs in enumerate(summaries, 1):
            # Show just the relative path or basename for cleaner prompt
            display_path = fs.file_id
            # Try to show path relative to common prefix (e.g., "data/enron/...")
            if '/' in fs.file_id:
                parts = fs.file_id.split('/')
                # Find "data" or similar and show from there
                if 'data' in parts:
                    idx = parts.index('data')
                    display_path = '/'.join(parts[idx:])
            
            summary_lines.append(f"{i}. {display_path}\n   {fs.global_summary}")
        
        summaries_text = "\n\n".join(summary_lines)
        
        prompt = f"""You are a file routing expert. You have access to {len(summaries)} files. Select EXACTLY the top {top_k} most relevant files for the given query.

Query: "{query}"

Available Files:
{summaries_text}

CRITICAL INSTRUCTIONS:
- You must select EXACTLY {top_k} files from the numbered list above (1-{len(summaries)})
- Return ONLY the numbers of the files, NOT file paths
- Do NOT invent file numbers outside the range 1-{len(summaries)}
- Order your selections by relevance (most relevant first)

Return a JSON object with this EXACT format:
{{
  "file_numbers": [1, 42, 17, ...]
}}

The file_numbers array must contain EXACTLY {top_k} numbers from the range 1-{len(summaries)}."""
        
        return prompt
    
    def _llm_route(self, prompt: str, top_k: int, summaries: List[FileSummary]) -> List[str]:
        """
        Call LLM to route query and parse response.
        
        Args:
            prompt: Formatted prompt with query and summaries
            top_k: Number of files to return
        
        Returns:
            List of routed file IDs
        """
        logger.info("Sending request to LLM...")
        
        # Use JSON mode to ensure valid JSON output
        response = litellm.completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic routing
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        logger.info("Received response from LLM")
        content = response.choices[0].message.content.strip()
        
        # Log raw response for debugging
        logger.info(f"Raw LLM response (first 500 chars): {content[:500]}")
        if len(content) > 500:
            logger.info(f"... (response truncated, total length: {len(content)} chars)")
        
        # Parse JSON response
        logger.info("Parsing LLM response...")
        file_ids = self._parse_llm_response(content, top_k, summaries)
        logger.info(f"Parse complete, extracted {len(file_ids)} file IDs")
        
        return file_ids
    
    def _parse_llm_response(self, response_text: str, top_k: int, summaries: List[FileSummary]) -> List[str]:
        """
        Parse LLM response to extract file IDs from file numbers.
        
        Args:
            response_text: Raw LLM response (guaranteed to be JSON due to JSON mode)
            top_k: Expected number of files
            summaries: List of FileSummary objects (for mapping numbers to IDs)
        
        Returns:
            List of file IDs
        """
        try:
            # With JSON mode, response should be clean JSON without markdown
            logger.info("Parsing JSON response...")
            data = json.loads(response_text)
            logger.info(f"JSON parsed successfully: {list(data.keys())}")
            
            if "file_numbers" in data:
                file_numbers = data["file_numbers"]
                logger.info(f"✓ Found 'file_numbers' key with {len(file_numbers)} entries")
                
                if len(file_numbers) > top_k:
                    logger.warning(f"LLM returned {len(file_numbers)} numbers, expected {top_k}. Truncating to {top_k}.")
                    file_numbers = file_numbers[:top_k]
                
                logger.info(f"LLM selected numbers (first 10): {file_numbers[:10]}")
                
                # Map numbers to file IDs
                logger.info("Mapping numbers to file IDs...")
                valid_ids = []
                for i, num in enumerate(file_numbers):
                    # Convert to 0-indexed
                    idx = num - 1
                    if 0 <= idx < len(summaries):
                        file_id = summaries[idx].file_id
                        valid_ids.append(file_id)
                        if i < 5:  # Log first 5 mappings
                            logger.info(f"  ✓ Number {num} → {file_id}")
                    else:
                        logger.warning(f"  ✗ Invalid number: {num} (valid range: 1-{len(summaries)})")
                
                logger.info(f"Mapping complete: {len(valid_ids)}/{len(file_numbers)} numbers mapped successfully")
                
                return valid_ids[:top_k]
            else:
                logger.error(f"✗ LLM response missing 'file_numbers' key!")
                logger.error(f"Available keys: {list(data.keys())}")
                logger.error(f"Full response data: {data}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"✗ JSON parsing failed!")
            logger.error(f"Error: {e}")
            logger.error(f"Failed text (first 1000 chars): {response_text[:1000]}")
            return []
    
    def _match_file_id(self, llm_file_id: str) -> Optional[str]:
        """
        Match LLM-returned file ID to a stored file ID.
        Handles various path formats (absolute, relative, basename).
        
        Args:
            llm_file_id: File ID returned by LLM
            
        Returns:
            Matching stored file ID, or None if no match
        """
        # Try exact match first
        if llm_file_id in self.file_summaries:
            return llm_file_id
        
        # Try matching as suffix (handles relative vs absolute paths)
        for stored_id in self.file_summaries.keys():
            # Check if stored path ends with LLM path
            if stored_id.endswith(llm_file_id):
                return stored_id
            # Check if LLM path ends with stored path (unlikely but possible)
            if llm_file_id.endswith(stored_id):
                return stored_id
        
        # Try basename matching as last resort
        llm_basename = os.path.basename(llm_file_id)
        for stored_id in self.file_summaries.keys():
            if os.path.basename(stored_id) == llm_basename:
                return stored_id
        
        return None
    
    def _embedding_fallback_route(
        self, 
        query: str, 
        summaries: List[FileSummary], 
        top_k: int
    ) -> List[str]:
        """
        Fallback routing using embedding similarity when context is too large.
        
        Args:
            query: The search query
            summaries: List of FileSummary objects
            top_k: Number of files to return
        
        Returns:
            List of file IDs sorted by embedding similarity
        """
        # This requires embeddings to be available
        # For now, just return first top_k files as a simple fallback
        logger.warning("Embedding fallback not yet implemented, returning first top_k files")
        return [fs.file_id for fs in summaries[:top_k]]
    
    def add_file_summary(self, file_summary: FileSummary) -> None:
        """
        Add a new file summary to the index.
        
        Args:
            file_summary: FileSummary object to add
        """
        self.file_summaries[file_summary.file_id] = file_summary
        logger.debug(f"Added file summary: {file_summary.file_id}")
    
    def remove_file(self, file_id: str) -> None:
        """
        Remove a file from the index.
        
        Args:
            file_id: ID of file to remove
        """
        if file_id in self.file_summaries:
            del self.file_summaries[file_id]
            logger.debug(f"Removed file: {file_id}")
    
    def get_summary(self, file_id: str) -> Optional[FileSummary]:
        """
        Get the summary for a specific file.
        
        Args:
            file_id: ID of file to get summary for
        
        Returns:
            FileSummary object or None if not found
        """
        return self.file_summaries.get(file_id)
    
    def __len__(self) -> int:
        """Return the number of files in the index."""
        return len(self.file_summaries)
