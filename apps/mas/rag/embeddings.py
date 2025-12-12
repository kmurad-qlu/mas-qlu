"""
Codestral Embeddings via OpenRouter.

Provides embedding functionality using mistralai/codestral-embed-2505 model
accessed through the OpenRouter API.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Union

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..infra.env import load_env, get_openrouter_api_key, get_openai_base_url, get_openrouter_headers


class CodestralEmbedder:
    """
    Embedding client using mistralai/codestral-embed-2505 via OpenRouter.
    
    This embedder provides both single query embedding and batch document
    embedding functionality, with automatic batching for large document sets.
    """
    
    # Codestral embedding model configuration
    MODEL = "mistralai/codestral-embed-2505"
    DIMENSION = 1536  # Codestral embedding dimension (verified via API)
    MAX_BATCH_SIZE = 32  # Maximum documents per API call
    MAX_INPUT_LENGTH = 8192  # Maximum tokens per input
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        batch_size: int = 32,
        request_timeout: int = 60,
    ):
        """
        Initialize the Codestral embedder.
        
        Args:
            api_key: OpenRouter API key. If None, reads from environment.
            batch_size: Number of documents to embed per API call.
            request_timeout: Timeout in seconds for API requests.
        """
        # Force reload env to pick up any changes
        from dotenv import load_dotenv
        load_dotenv(override=True)
        load_env()
        
        self.api_key = api_key or get_openrouter_api_key()
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY.")
        
        # Strip any whitespace from key
        self.api_key = self.api_key.strip()
        
        base_url = get_openai_base_url()
        headers = get_openrouter_headers()
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            default_headers=headers,
        )
        self.batch_size = min(batch_size, self.MAX_BATCH_SIZE)
        self.request_timeout = request_timeout
        self.dimension = self.DIMENSION
    
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed (max batch_size)
        
        Returns:
            numpy array of shape (len(texts), DIMENSION)
        """
        if not texts:
            return np.array([]).reshape(0, self.DIMENSION)
        
        # Truncate texts that are too long
        truncated_texts = []
        for text in texts:
            # Rough estimate: 4 chars per token
            if len(text) > self.MAX_INPUT_LENGTH * 4:
                truncated_texts.append(text[:self.MAX_INPUT_LENGTH * 4])
            else:
                truncated_texts.append(text)
        
        response = self.client.embeddings.create(
            model=self.MODEL,
            input=truncated_texts,
            timeout=self.request_timeout,
        )
        
        # Extract embeddings in order
        embeddings = []
        for item in sorted(response.data, key=lambda x: x.index):
            embeddings.append(item.embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            query: The query text to embed
        
        Returns:
            numpy array of shape (DIMENSION,)
        """
        if not query or not query.strip():
            return np.zeros(self.DIMENSION, dtype=np.float32)
        
        result = self._embed_batch([query.strip()])
        return result[0]
    
    def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed multiple documents with automatic batching.
        
        Args:
            texts: List of document texts to embed
            show_progress: Whether to print progress
        
        Returns:
            numpy array of shape (len(texts), DIMENSION)
        """
        if not texts:
            return np.array([]).reshape(0, self.DIMENSION)
        
        # Filter empty texts but track indices
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text.strip())
        
        if not non_empty_texts:
            return np.zeros((len(texts), self.DIMENSION), dtype=np.float32)
        
        # Process in batches
        all_embeddings = []
        num_batches = (len(non_empty_texts) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(non_empty_texts))
            batch = non_empty_texts[start:end]
            
            if show_progress:
                print(f"  Embedding batch {batch_idx + 1}/{num_batches} ({len(batch)} docs)...")
            
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.append(batch_embeddings)
            
            # Rate limiting: small delay between batches
            if batch_idx < num_batches - 1:
                time.sleep(0.1)
        
        # Concatenate all batch results
        non_empty_embeddings = np.vstack(all_embeddings)
        
        # Reconstruct full array with zeros for empty texts
        result = np.zeros((len(texts), self.DIMENSION), dtype=np.float32)
        for i, orig_idx in enumerate(non_empty_indices):
            result[orig_idx] = non_empty_embeddings[i]
        
        return result
    
    def embed_documents_iter(
        self,
        texts: List[str],
        show_progress: bool = False,
    ):
        """
        Generator that yields embeddings batch by batch.
        
        Useful for large document sets where you want to process
        results incrementally.
        
        Args:
            texts: List of document texts to embed
            show_progress: Whether to print progress
        
        Yields:
            Tuple of (batch_indices, batch_embeddings)
        """
        if not texts:
            return
        
        # Filter empty texts
        non_empty = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
        
        if not non_empty:
            return
        
        num_batches = (len(non_empty) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(non_empty))
            batch_items = non_empty[start:end]
            
            indices = [item[0] for item in batch_items]
            batch_texts = [item[1] for item in batch_items]
            
            if show_progress:
                print(f"  Embedding batch {batch_idx + 1}/{num_batches}...")
            
            embeddings = self._embed_batch(batch_texts)
            yield indices, embeddings
            
            if batch_idx < num_batches - 1:
                time.sleep(0.1)

