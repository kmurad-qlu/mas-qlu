"""
Text chunking strategies for RAG ingestion.

Provides utilities to split documents into chunks suitable for embedding
while preserving semantic coherence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    start_char: int
    end_char: int
    chunk_idx: int


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    min_chunk_size: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk (in characters)
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size (avoid tiny trailing chunks)
    
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to break at sentence boundary if not at the end
        if end < text_len:
            # Look for sentence endings within the last 20% of the chunk
            search_start = start + int(chunk_size * 0.8)
            search_region = text[search_start:end]
            
            # Find last sentence boundary (. ! ? followed by space or newline)
            sentence_breaks = list(re.finditer(r'[.!?]\s', search_region))
            if sentence_breaks:
                # Use the last sentence break found
                last_break = sentence_breaks[-1]
                end = search_start + last_break.end()
            else:
                # Try to break at word boundary
                last_space = text.rfind(' ', start + int(chunk_size * 0.8), end)
                if last_space > start:
                    end = last_space + 1
        
        chunk = text[start:end].strip()
        if chunk and len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        elif chunk and chunks:
            # Append small trailing chunk to previous
            chunks[-1] = chunks[-1] + " " + chunk
        elif chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < text_len else text_len
        
        # Avoid infinite loop
        if start <= 0 or (end >= text_len and len(chunks) > 0):
            if end >= text_len:
                break
            start = max(1, end - overlap)
    
    return chunks


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    min_chunk_size: int = 100,
) -> List[Chunk]:
    """
    Split text into overlapping chunks with position metadata.
    
    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk (in characters)
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size
    
    Returns:
        List of Chunk objects with position information
    """
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [Chunk(text=text, start_char=0, end_char=len(text), chunk_idx=0)]
    
    chunks = []
    start = 0
    text_len = len(text)
    chunk_idx = 0
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to break at sentence boundary
        if end < text_len:
            search_start = start + int(chunk_size * 0.8)
            search_region = text[search_start:end]
            sentence_breaks = list(re.finditer(r'[.!?]\s', search_region))
            if sentence_breaks:
                last_break = sentence_breaks[-1]
                end = search_start + last_break.end()
            else:
                last_space = text.rfind(' ', start + int(chunk_size * 0.8), end)
                if last_space > start:
                    end = last_space + 1
        
        chunk_text_content = text[start:end].strip()
        if chunk_text_content and len(chunk_text_content) >= min_chunk_size:
            chunks.append(Chunk(
                text=chunk_text_content,
                start_char=start,
                end_char=end,
                chunk_idx=chunk_idx,
            ))
            chunk_idx += 1
        
        start = end - overlap if end < text_len else text_len
        if start <= 0 or end >= text_len:
            break
    
    return chunks


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    url: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[dict]:
    """
    Chunk a document and return list of chunk dictionaries ready for indexing.
    
    Args:
        doc_id: Document identifier
        title: Document title
        text: Document text content
        url: Document URL
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
    
    Returns:
        List of dictionaries with chunk data
    """
    chunks = chunk_text_with_metadata(text, chunk_size=chunk_size, overlap=overlap)
    
    result = []
    for chunk in chunks:
        # Prepend title to chunk for better context
        chunk_with_title = f"{title}\n\n{chunk.text}" if title else chunk.text
        
        result.append({
            "id": f"{doc_id}_chunk_{chunk.chunk_idx}",
            "doc_id": doc_id,
            "title": title,
            "text": chunk_with_title,
            "url": url,
            "chunk_idx": chunk.chunk_idx,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        })
    
    return result

