"""
Text Processor Module
Handles text chunking, preprocessing, and manipulation
"""

import re
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from logging_config import get_logger
from config import (
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_MAX,
    CHUNK_OVERLAP
)

logger = get_logger(__name__)


class ChunkingStrategy(Enum):
    """Different strategies for chunking text"""
    WORD_BOUNDARY = "word_boundary"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BASED = "token_based"


@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    text: str
    index: int
    start_pos: int
    end_pos: int
    word_count: int
    char_count: int
    metadata: Dict[str, Any] = None
    
    @property
    def is_complete_sentence(self) -> bool:
        """Check if chunk ends with sentence terminator"""
        return self.text.rstrip().endswith(('.', '!', '?'))


@dataclass
class ChunkingResult:
    """Result of text chunking operation"""
    chunks: List[TextChunk]
    total_chunks: int
    strategy_used: ChunkingStrategy
    average_chunk_size: float
    overlap_used: int
    metadata: Dict[str, Any] = None


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
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for i, word in enumerate(words):
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.target_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    word_count=len(current_chunk),
                    char_count=len(chunk_text)
                ))
                
                # Handle overlap
                if self.overlap > 0:
                    # Keep last N characters worth of words for overlap
                    overlap_words = []
                    overlap_length = 0
                    for w in reversed(current_chunk):
                        overlap_length += len(w) + 1
                        if overlap_length >= self.overlap:
                            break
                        overlap_words.insert(0, w)
                    current_chunk = overlap_words
                    current_length = sum(len(w) + 1 for w in overlap_words)
                    start_pos = start_pos + len(chunk_text) - current_length
                else:
                    current_chunk = []
                    current_length = 0
                    start_pos += len(chunk_text) + 1
                    
            current_chunk.append(word)
            current_length += word_length
            
        # Add remaining words
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                word_count=len(current_chunk),
                char_count=len(chunk_text)
            ))
            
        return chunks
        
    def _chunk_by_sentences(self, text: str) -> List[TextChunk]:
        """Chunk text at sentence boundaries"""
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 1
            
            if current_length + sentence_length > self.target_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text)
                ))
                
                # Handle overlap (keep last sentence if overlap enabled)
                if self.overlap > 0 and len(current_chunk) > 1:
                    overlap_sentence = current_chunk[-1]
                    current_chunk = [overlap_sentence]
                    current_length = len(overlap_sentence)
                    start_pos = start_pos + len(chunk_text) - current_length
                else:
                    current_chunk = []
                    current_length = 0
                    start_pos += len(chunk_text) + 1
                    
            current_chunk.append(sentence)
            current_length += sentence_length
            
        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            ))
            
        return chunks
        
    def _chunk_by_paragraphs(self, text: str) -> List[TextChunk]:
        """Chunk text at paragraph boundaries"""
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para) + 2  # +2 for double newline
            
            if current_length + para_length > self.target_size and current_chunk:
                # Create chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text)
                ))
                
                current_chunk = []
                current_length = 0
                start_pos += len(chunk_text) + 2
                
            # If single paragraph is too large, split it
            if para_length > self.max_size:
                # Recursively chunk the paragraph
                sub_chunker = TextChunker(
                    target_size=self.target_size,
                    strategy=ChunkingStrategy.SENTENCE_BOUNDARY
                )
                sub_result = sub_chunker.chunk_text(para)
                
                for sub_chunk in sub_result.chunks:
                    chunks.append(TextChunk(
                        text=sub_chunk.text,
                        index=len(chunks),
                        start_pos=start_pos,
                        end_pos=start_pos + sub_chunk.char_count,
                        word_count=sub_chunk.word_count,
                        char_count=sub_chunk.char_count
                    ))
                    start_pos += sub_chunk.char_count + 1
            else:
                current_chunk.append(para)
                current_length += para_length
                
        # Add remaining paragraphs
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            ))
            
        return chunks
        
    def _chunk_semantically(self, text: str) -> List[TextChunk]:
        """
        Chunk text based on semantic similarity (simplified version)
        In production, this would use embeddings and clustering
        """
        # For now, fall back to paragraph chunking with topic detection
        # This is a placeholder for more sophisticated semantic chunking
        logger.debug("Semantic chunking not fully implemented, using paragraph strategy")
        return self._chunk_by_paragraphs(text)
        
    def _chunk_sliding_window(self, text: str) -> List[TextChunk]:
        """Chunk text using sliding window approach"""
        chunks = []
        step_size = self.target_size - self.overlap
        
        for i in range(0, len(text), step_size):
            chunk_text = text[i:i + self.target_size]
            
            # Skip if chunk is too small (except last chunk)
            if len(chunk_text) < self.min_size and i + self.target_size < len(text):
                continue
                
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                start_pos=i,
                end_pos=i + len(chunk_text),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            ))
            
        return chunks
        
    def _chunk_by_tokens(self, text: str) -> List[TextChunk]:
        """
        Chunk text by approximate token count
        (Simplified - assumes ~4 chars per token)
        """
        chars_per_token = 4
        target_chars = self.target_size
        
        # Convert to approximate token boundaries
        return self._chunk_by_words(text)
        
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
        self.logger = get_logger(f"{__name__}.Preprocessor")
        
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
