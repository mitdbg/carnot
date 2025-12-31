import os
import time
import uuid
from typing import List, Any
import tiktoken
import chromadb.utils.embedding_functions as embedding_functions

from carnot.core.data.smv import Chunk, ChunkIndex, FileSummary, TaggedFiles, MetadataRegistry
from carnot.constants import Model

class SMVGenerator:
    def __init__(self):
        # Initialize embedding function
        # TODO: Make this configurable/consistent with ContextManager
        self.emb_fn = None
        if os.getenv("OPENAI_API_KEY"):
            self.emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate_metadata_registry(self, file_id: str, file_path: str, file_content: bytes | str) -> MetadataRegistry:
        """Generates the MetadataRegistry SMV."""
        
        # Determine file size
        if isinstance(file_content, str):
            size = len(file_content.encode('utf-8'))
        else:
            size = len(file_content)
            
        # Determine file type
        _, ext = os.path.splitext(file_path)
        file_type = ext.lower().lstrip('.')
        
        # Get creation time
        try:
            creation_time = time.ctime(os.path.getctime(file_path))
        except Exception:
            creation_time = time.ctime()

        # Simple heuristic for quality score (e.g., based on length for now)
        # TODO: Improve this heuristic
        quality_score = min(1.0, size / 10000.0) 

        return MetadataRegistry(
            file_id=file_id,
            file_size_bytes=size,
            file_type=file_type,
            creation_date=creation_time,
            topic_clusters=[], # To be filled by a separate process or LLM
            quality_score=quality_score
        )

    def generate_chunk_index(self, file_id: str, text_content: str, chunk_size_tokens: int = 512, overlap_tokens: int = 50) -> ChunkIndex:
        """Generates the ChunkIndex SMV by splitting text and embedding chunks."""
        
        tokens = self.tokenizer.encode(text_content)
        total_tokens = len(tokens)
        
        chunks: List[Chunk] = []
        
        start_idx = 0
        while start_idx < total_tokens:
            end_idx = min(start_idx + chunk_size_tokens, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Find character offsets (approximation as tiktoken doesn't give char offsets directly easily)
            # For exact offsets, we'd need to map back. For now, we'll search in the text.
            # This is a bit expensive and potentially buggy if duplicates exist. 
            # A better approach for production is to track offsets during tokenization or use a different chunker.
            # For this MVP, we will rely on string search from the last position.
            
            # TODO: Implement robust character offset tracking.
            # For now, we will set them to -1 or approximate if needed, but let's try to find it.
            # Actually, let's just use the text for now.
            
            chunk_obj = Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                start_char_idx=-1, # Placeholder
                end_char_idx=-1,   # Placeholder
                embedding=None
            )
            chunks.append(chunk_obj)
            
            start_idx += (chunk_size_tokens - overlap_tokens)
        
        # Batch embed chunks
        if self.emb_fn and chunks:
            texts = [c.text for c in chunks]
            embeddings = self.emb_fn(texts)
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]

        return ChunkIndex(
            file_id=file_id,
            chunks=chunks,
            model_name="text-embedding-3-small" # Hardcoded for now
        )

    def generate_file_summary(self, file_id: str, text_content: str, file_path: str = None) -> FileSummary:
        """Generates the FileSummary SMV using an LLM."""
        import litellm
        import os
        
        # Truncate very long files to fit in context
        # Keep first portion of file for summary generation
        max_chars = 50000  # ~12k tokens roughly
        if len(text_content) > max_chars:
            text_content = text_content[:max_chars] + "\n\n[... content truncated ...]"
        
        # Build prompt optimized for routing
        prompt = f"""Analyze the following file and create a concise summary optimized for query routing.

File: {file_path or file_id}

Content:
{text_content}

Generate a 2-3 sentence summary that:
1. Identifies the main topics/themes in the file
2. Highlights key entities, dates, or events mentioned
3. Uses specific terms that would match relevant queries

Summary:"""
        
        # Choose model
        model_id = "openai/gpt-4o-mini"
        if os.getenv("ANTHROPIC_API_KEY"):
            model_id = "anthropic/claude-3-5-sonnet-20241022"
        elif os.getenv("GEMINI_API_KEY"):
            model_id = "vertex_ai/gemini-2.0-flash-001"
        
        try:
            response = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            summary_text = response.choices[0].message.content.strip()
            
            # Generate embedding of summary if embedding function available
            summary_embedding = None
            if self.emb_fn:
                summary_embedding = self.emb_fn([summary_text])[0]
            
            return FileSummary(
                file_id=file_id,
                global_summary=summary_text,
                dense_blocks=[],  # Could extract key sections in future
                summary_embedding=summary_embedding
            )
            
        except Exception as e:
            # Fallback to simple summary on error
            print(f"Failed to generate LLM summary for {file_id}: {e}")
            
            # Create simple fallback summary
            preview = text_content[:500].replace("\n", " ")
            fallback_summary = f"File containing: {preview}..."
            
            return FileSummary(
                file_id=file_id,
                global_summary=fallback_summary,
                dense_blocks=[],
                summary_embedding=None
            )




