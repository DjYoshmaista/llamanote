"""
Text Processor Module
Handles text chunking, preprocessing, and manipulation
"""

import re
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..utils.logger import get_logger_conf
from ..core.types import ChunkingStrategy, TextChunk, ChunkingResult
from ..config.settings import (
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_MAX,
    CHUNK_OVERLAP
)

logger = get_logger_conf(__name__)

def _apply_overlap_and_create_chunks(self, 
                                     initial_chunks: List[str], 
                                     strategy: ChunkingStrategy) -> List[TextChunk]:
    """
    Helper method to apply overlap and create TextChunk objects.
    
    Args:
        initial_chunks: List of raw text chunks
        strategy: The chunking strategy used
        
    Returns:
        List of TextChunk objects with overlap applied
    """
    if not initial_chunks:
        return []
    
    final_chunks = []
    
    for i, chunk_text in enumerate(initial_chunks):
        # Apply overlap from previous chunk
        if i > 0 and self.overlap_size > 0:
            prev_chunk = initial_chunks[i - 1]
            overlap_text = prev_chunk[-self.overlap_size:] if len(prev_chunk) > self.overlap_size else prev_chunk
            chunk_text = overlap_text + " " + chunk_text
        
        # Create TextChunk object
        text_chunk = TextChunk(
            text=chunk_text,
            index=i,
            char_count=len(chunk_text),
            word_count=len(chunk_text.split()),
            strategy=strategy
        )
        final_chunks.append(text_chunk)
    
    return final_chunks

class TextChunker:
    """Advanced text chunking with multiple strategies"""
    
    def __init__(self,
                 target_size: int = CHUNK_SIZE_DEFAULT,
                 min_size: int = CHUNK_SIZE_MIN,
                 max_size: int = CHUNK_SIZE_MAX,
                 overlap: int = CHUNK_OVERLAP,
                 strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY):
        """
        Initialize text chunker
        
        Args:
            target_size: Target chunk size in characters
            min_size: Minimum chunk size
            max_size: Maximum chunk size
            overlap: Characters to overlap between chunks
            strategy: Chunking strategy to use
        """
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap
        self.strategy = strategy
        
        logger.info(f"Initialized TextChunker (strategy={strategy.value}, "
                   f"target_size={target_size}, overlap={overlap})")

    def _chunk_with_accumulator(self,
                                items: List[str],
                                item_sizes: List[int],
                                join_str: str = ' ') -> List[TextChunk]:
        """
    Generic chunking logic using accumulator pattern.
        
        Args:
            items: List of text items to chunk (words, sentences, paragraphs)
            item_sizes: List of sizes for each item
            join_str: String to join items with
        
        Returns:
            List of TextChunk objects
        """
        chunks = []
        current_items = []
        current_length = 0
        start_pos = 0
        
        for i, (item, item_size) in enumerate(zip(items, item_sizes)):
            # Check if adding this item would exceed target
            if current_length + item_size > self.target_size and current_items:
                # Create chunk from accumulated items
                chunk_text = join_str.join(current_items)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text)
                ))
                
                # Handle overlap
                if self.overlap > 0 and len(current_items) > 1:
                    # Keep items for overlap
                    overlap_items = []
                    overlap_length = 0
                    for idx in range(len(current_items) - 1, -1, -1):
                        overlap_length += item_sizes[i - len(current_items) + idx]
                        if overlap_length >= self.overlap:
                            break
                        overlap_items.insert(0, current_items[idx])
                    
                    current_items = overlap_items
                    current_length = sum(len(item) + len(join_str) for item in current_items)
                    start_pos = start_pos + len(chunk_text) - current_length
                else:
                    current_items = []
                    current_length = 0
                    start_pos += len(chunk_text) + len(join_str)
            
            current_items.append(item)
            current_length += item_size
        
        # Add remaining items
        if current_items:
            chunk_text = join_str.join(current_items)
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            ))
        
        return chunks
    
    def chunk_text(self, text: str, 
                  strategy: Optional[ChunkingStrategy] = None) -> ChunkingResult:
        """
        Chunk text using specified or default strategy
        
        Args:
            text: Text to chunk
            strategy: Override default strategy
            
        Returns:
            ChunkingResult with chunks and metadata
        """
        strategy = strategy or self.strategy
        
        logger.debug(f"Chunking text of length {len(text)} using {strategy.value}")
        
        if strategy == ChunkingStrategy.WORD_BOUNDARY:
            chunks = self._chunk_by_words(text)
        elif strategy == ChunkingStrategy.SENTENCE_BOUNDARY:
            chunks = self._chunk_by_sentences(text)
        elif strategy == ChunkingStrategy.PARAGRAPH_BOUNDARY:
            chunks = self._chunk_by_paragraphs(text)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantically(text)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            chunks = self._chunk_sliding_window(text)
        elif strategy == ChunkingStrategy.TOKEN_BASED:
            chunks = self._chunk_by_tokens(text)
        else:
            chunks = self._chunk_by_words(text)  # Fallback
            
        # Calculate statistics
        total_chars = sum(c.char_count for c in chunks)
        avg_size = total_chars / len(chunks) if chunks else 0
        
        logger.info(f"Created {len(chunks)} chunks, avg size: {avg_size:.0f} chars")
        
        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            strategy_used=strategy,
            average_chunk_size=avg_size,
            overlap_used=self.overlap,
            metadata={"original_length": len(text)}
        )

    def _chunk_by_words(self, text: str) -> List[TextChunk]:
        """Chunk text at word boundaries"""
        words = text.split()
        sizes = [len(word) + 1 for word in words]  # +1 for space
        return self._chunk_with_accumulator(words, sizes, join_str=' ')
    
    def _chunk_by_sentences(self, text: str) -> List[TextChunk]:
        """Chunk text at sentence boundaries"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sizes = [len(sent) + 1 for sent in sentences]  # +1 for space
        return self._chunk_with_accumulator(sentences, sizes, join_str=' ')

    def _chunk_by_paragraphs(self, text: str) -> List[TextChunk]:
        """Chunk by paragraphs, combining small ones to reach target size."""
        # Split into paragraphs (double newline or single newline followed by indent)
        paragraphs = re.split(r'\n\s*\n|\n(?=\s{4,})', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Combine small paragraphs
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.target_size:  # +2 for newlines
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return self._apply_overlap_and_create_chunks(chunks, ChunkingStrategy.PARAGRAPH)

    def _chunk_semantically(self, text: str) -> List[TextChunk]:
        """Chunk by semantic units (sentences grouped by topic similarity)."""
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence exceeds target
            if current_size + sentence_size > self.target_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return self._apply_overlap_and_create_chunks(chunks, ChunkingStrategy.SEMANTIC)

    def _chunk_by_sliding_window(self, text: str) -> List[TextChunk]:
        """Create overlapping chunks using a sliding window approach."""
        if len(text) <= self.target_size:
            return self._apply_overlap_and_create_chunks([text], ChunkingStrategy.SLIDING_WINDOW)
        
        chunks = []
        stride = self.target_size - self.overlap_size  # Step size
        
        if stride <= 0:
            stride = self.target_size // 2  # Fallback to 50% overlap
        
        position = 0
        while position < len(text):
            end_position = min(position + self.target_size, len(text))
            chunk = text[position:end_position]
            chunks.append(chunk)
            
            if end_position >= len(text):
                break
                
            position += stride
        
        # Note: overlap is already built into the sliding window, so we pass it
        # directly but the helper will add extra overlap between chunks
        return self._apply_overlap_and_create_chunks(chunks, ChunkingStrategy.SLIDING_WINDOW)

    def _chunk_by_tokens(self, text: str) -> List[TextChunk]:
        """Chunk by approximate token count (rough estimation)."""
        # Rough token estimate: ~4 chars per token
        chars_per_token = 4
        target_chars = self.target_size * chars_per_token
        
        # Split by sentences for cleaner boundaries
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > target_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return self._apply_overlap_and_create_chunks(chunks, ChunkingStrategy.TOKEN_BASED)
           
    def merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge chunks that are too small"""
        merged = []
        buffer = []
        buffer_size = 0
        
        for chunk in chunks:
            if chunk.char_count < self.min_size:
                buffer.append(chunk)
                buffer_size += chunk.char_count
                
                # Check if buffer is large enough
                if buffer_size >= self.target_size:
                    # Merge buffer chunks
                    merged_text = ' '.join(c.text for c in buffer)
                    merged.append(TextChunk(
                        text=merged_text,
                        index=len(merged),
                        start_pos=buffer[0].start_pos,
                        end_pos=buffer[-1].end_pos,
                        word_count=sum(c.word_count for c in buffer),
                        char_count=len(merged_text)
                    ))
                    buffer = []
                    buffer_size = 0
            else:
                # Add any buffered chunks first
                if buffer:
                    if merged and buffer_size < self.min_size // 2:
                        # Merge with previous chunk
                        prev = merged[-1]
                        merged_text = prev.text + ' ' + ' '.join(c.text for c in buffer)
                        merged[-1] = TextChunk(
                            text=merged_text,
                            index=prev.index,
                            start_pos=prev.start_pos,
                            end_pos=buffer[-1].end_pos,
                            word_count=prev.word_count + sum(c.word_count for c in buffer),
                            char_count=len(merged_text)
                        )
                    else:
                        # Create new chunk from buffer
                        merged_text = ' '.join(c.text for c in buffer)
                        merged.append(TextChunk(
                            text=merged_text,
                            index=len(merged),
                            start_pos=buffer[0].start_pos,
                            end_pos=buffer[-1].end_pos,
                            word_count=sum(c.word_count for c in buffer),
                            char_count=len(merged_text)
                        ))
                    buffer = []
                    buffer_size = 0
                    
                # Add current chunk
                merged.append(chunk)
                
        # Handle remaining buffer
        if buffer:
            if merged:
                # Merge with last chunk
                prev = merged[-1]
                merged_text = prev.text + ' ' + ' '.join(c.text for c in buffer)
                merged[-1] = TextChunk(
                    text=merged_text,
                    index=prev.index,
                    start_pos=prev.start_pos,
                    end_pos=buffer[-1].end_pos,
                    word_count=prev.word_count + sum(c.word_count for c in buffer),
                    char_count=len(merged_text)
                )
            else:
                # Create final chunk
                merged_text = ' '.join(c.text for c in buffer)
                merged.append(TextChunk(
                    text=merged_text,
                    index=0,
                    start_pos=buffer[0].start_pos,
                    end_pos=buffer[-1].end_pos,
                    word_count=sum(c.word_count for c in buffer),
                    char_count=len(merged_text)
                ))
                
        return merged


class TextPreprocessor:
    """Preprocess text for different purposes"""
    
    def __init__(self):
        self.logger = get_logger_conf(f"{__name__}.Preprocessor")
        
    def preprocess_for_llm(self, 
                          text: str,
                          remove_urls: bool = True,
                          remove_emails: bool = True,
                          normalize_whitespace: bool = True,
                          fix_encoding: bool = True) -> str:
        """
        Preprocess text for LLM processing
        
        Args:
            text: Input text
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize_whitespace: Whether to normalize whitespace
            fix_encoding: Whether to fix encoding issues
            
        Returns:
            Preprocessed text
        """
        if fix_encoding:
            text = self._fix_encoding_issues(text)
            
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
            
        if remove_emails:
            text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
            
        if normalize_whitespace:
            text = self._normalize_whitespace(text)
            
        return text
        
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues"""
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '--',
            '…': '...',
            '\u200b': '',  # Zero-width space
            '\ufeff': '',  # BOM
            '\xa0': ' ',   # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Remove other non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        return text
        
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace tabs with spaces
        text = text.replace('\t', '    ')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # Remove trailing whitespace
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Reduce multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Reduce multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
