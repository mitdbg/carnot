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
        if not self.file_summaries:
            logger.warning("No file summaries available for routing")
            return []
        
        # Limit files if specified
        summaries_to_use = list(self.file_summaries.values())
        if max_files and len(summaries_to_use) > max_files:
            logger.info(f"Limiting routing to {max_files} files (total: {len(summaries_to_use)})")
            summaries_to_use = summaries_to_use[:max_files]
        
        # Build prompt
        prompt = self._format_summaries_for_llm(query, summaries_to_use, top_k)
        
        # Check if prompt fits in context
        prompt_tokens = len(self.tokenizer.encode(prompt))
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
        
        logger.info(f"Routing query with {prompt_tokens} tokens to LLM")
        
        # Call LLM for routing
        try:
            routed_ids = self._llm_route(prompt, top_k)
            logger.info(f"LLM routed to {len(routed_ids)} files: {routed_ids[:5]}...")
            return routed_ids
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
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
        # Build numbered list of summaries
        summary_lines = []
        for i, fs in enumerate(summaries, 1):
            summary_lines.append(f"{i}. [{fs.file_id}] {fs.global_summary}")
        
        summaries_text = "\n".join(summary_lines)
        
        prompt = f"""You are a file routing expert. Given a query and a list of file summaries, select the {top_k} most relevant files.

Query: "{query}"

File Summaries:
{summaries_text}

Analyze the query and file summaries carefully. Return ONLY a JSON object with a "file_ids" array containing the file IDs of the {top_k} most relevant files, ordered by relevance (most relevant first).

Response format: {{"file_ids": ["file_id_1", "file_id_2", ...]}}

JSON Response:"""
        
        return prompt
    
    def _llm_route(self, prompt: str, top_k: int) -> List[str]:
        """
        Call LLM to route query and parse response.
        
        Args:
            prompt: Formatted prompt with query and summaries
            top_k: Number of files to return
        
        Returns:
            List of routed file IDs
        """
        response = litellm.completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Deterministic routing
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        file_ids = self._parse_llm_response(content, top_k)
        
        return file_ids
    
    def _parse_llm_response(self, response_text: str, top_k: int) -> List[str]:
        """
        Parse LLM response to extract file IDs.
        
        Args:
            response_text: Raw LLM response
            top_k: Expected number of files
        
        Returns:
            List of file IDs
        """
        try:
            # Try to parse as JSON
            # Handle case where LLM adds markdown code blocks
            if "```json" in response_text:
                # Extract JSON from code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                # Generic code block
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            data = json.loads(response_text)
            
            if "file_ids" in data:
                file_ids = data["file_ids"]
                # Validate file IDs exist in our index
                valid_ids = [fid for fid in file_ids if fid in self.file_summaries]
                
                if len(valid_ids) < len(file_ids):
                    logger.warning(
                        f"LLM returned {len(file_ids) - len(valid_ids)} invalid file IDs"
                    )
                
                return valid_ids[:top_k]
            else:
                logger.error(f"LLM response missing 'file_ids' key: {data}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}")
            return []
    
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
