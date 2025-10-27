"""
Response Filter Module
Filters out thinking portions and unwanted patterns from LLM responses
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from logging_config import get_logger
from config import THINKING_PATTERNS, ModelConfig

logger = get_logger(__name__)


@dataclass
class FilterResult:
    """Result of filtering operation"""
    original_text: str
    filtered_text: str
    removed_segments: List[str]
    filter_stats: Dict[str, int]
    
    @property
    def removal_ratio(self) -> float:
        """Calculate the ratio of text removed"""
        if len(self.original_text) == 0:
            return 0.0
        return 1.0 - (len(self.filtered_text) / len(self.original_text))


class ResponseFilter:
    """Filters thinking tokens and unwanted patterns from LLM responses"""
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.model_config = model_config
        self.custom_patterns: List[Tuple[str, str]] = []
        self.thinking_patterns = THINKING_PATTERNS.copy()
        
        # Add model-specific thinking tokens if provided
        if model_config and model_config.thinking_tokens:
            for token in model_config.thinking_tokens:
                # Escape special regex characters
                escaped_token = re.escape(token)
                # Create pattern to match content between paired tokens
                if token.endswith(">") or token.endswith("]"):
                    # Assume paired tokens
                    start_token = escaped_token
                    end_token = escaped_token.replace("<", "</").replace("[", "[/")
                    pattern = f"{start_token}(.*?){end_token}"
                    self.thinking_patterns.append((pattern, ""))
                    
        logger.info(f"Initialized ResponseFilter with {len(self.thinking_patterns)} patterns")
        
    def add_custom_pattern(self, pattern: str, replacement: str = ""):
        """Add a custom filtering pattern"""
        self.custom_patterns.append((pattern, replacement))
        logger.debug(f"Added custom pattern: {pattern}")
        
    def filter(self, text: str, 
              remove_thinking: bool = True,
              remove_acknowledgments: bool = True,
              preserve_structure: bool = True) -> FilterResult:
        """
        Filter unwanted content from text
        
        Args:
            text: Input text to filter
            remove_thinking: Remove thinking patterns
            remove_acknowledgments: Remove AI acknowledgments
            preserve_structure: Preserve paragraph structure
            
        Returns:
            FilterResult object containing filtered text and statistics
        """
        logger.debug(f"Starting filter on text of length {len(text)}")
        
        original_text = text
        filtered_text = text
        removed_segments = []
        filter_stats = {}
        
        # Remove thinking patterns
        if remove_thinking:
            filtered_text, thinking_removed = self._remove_patterns(
                filtered_text, 
                self.thinking_patterns + self.custom_patterns,
                "thinking"
            )
            removed_segments.extend(thinking_removed)
            filter_stats['thinking_segments'] = len(thinking_removed)
            
        # Remove acknowledgment patterns
        if remove_acknowledgments:
            acknowledgment_patterns = [
                (r"^(Sure|Certainly|Of course|I understand|I'll|Let me)[\s,!.]+", ""),
                (r"^Here's? (the|your|a) (cleaned|processed|formatted).*?:\n+", ""),
                (r"^I've (processed|cleaned|formatted).*?\n+", ""),
            ]
            filtered_text, ack_removed = self._remove_patterns(
                filtered_text,
                acknowledgment_patterns,
                "acknowledgment"
            )
            removed_segments.extend(ack_removed)
            filter_stats['acknowledgment_segments'] = len(ack_removed)
            
        # Clean up whitespace while preserving structure
        if preserve_structure:
            filtered_text = self._preserve_structure(filtered_text)
        else:
            # Aggressive whitespace cleaning
            filtered_text = re.sub(r'\n{3,}', '\n\n', filtered_text)
            filtered_text = re.sub(r' {2,}', ' ', filtered_text)
            
        # Remove any remaining special tokens
        filtered_text = self._remove_special_tokens(filtered_text)
        
        # Final cleanup
        filtered_text = filtered_text.strip()
        
        # Log statistics
        logger.info(
            f"Filtering complete: Original={len(original_text)} chars, "
            f"Filtered={len(filtered_text)} chars, "
            f"Removed={len(original_text) - len(filtered_text)} chars "
            f"({(1 - len(filtered_text)/len(original_text)) * 100:.1f}%)"
        )
        
        return FilterResult(
            original_text=original_text,
            filtered_text=filtered_text,
            removed_segments=removed_segments,
            filter_stats=filter_stats
        )
        
    def _remove_patterns(self, text: str, patterns: List[Tuple[str, str]], 
                        pattern_type: str) -> Tuple[str, List[str]]:
        """Remove patterns from text and track what was removed"""
        removed_segments = []
        
        for pattern, replacement in patterns:
            try:
                # Find all matches before removing
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    # Store removed content
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]  # Get first group if multiple groups
                        if match and len(match.strip()) > 0:
                            removed_segments.append(f"[{pattern_type}]: {match[:100]}...")
                            
                    # Remove the pattern
                    text = re.sub(pattern, replacement, text, flags=re.DOTALL | re.IGNORECASE)
                    logger.debug(f"Removed {len(matches)} instances of pattern: {pattern[:50]}...")
                    
            except Exception as e:
                logger.warning(f"Failed to apply pattern {pattern}: {e}")
                
        return text, removed_segments
        
    def _preserve_structure(self, text: str) -> str:
        """Clean up whitespace while preserving document structure"""
        # Preserve paragraph breaks (double newlines)
        paragraphs = text.split('\n\n')
        
        cleaned_paragraphs = []
        for para in paragraphs:
            # Clean up each paragraph
            para = para.strip()
            if para:
                # Replace multiple spaces with single space
                para = re.sub(r' {2,}', ' ', para)
                # Replace multiple newlines within paragraph with single newline
                para = re.sub(r'\n{2,}', '\n', para)
                cleaned_paragraphs.append(para)
                
        # Rejoin paragraphs with double newlines
        return '\n\n'.join(cleaned_paragraphs)
        
    def _remove_special_tokens(self, text: str) -> str:
        """Remove any remaining special tokens that might have been missed"""
        special_tokens = [
            '<|endoftext|>',
            '<|end|>',
            '<|assistant|>',
            '<|system|>',
            '<|user|>',
            '[INST]',
            '[/INST]',
            '<<SYS>>',
            '<</SYS>>',
            '<s>',
            '</s>'
        ]
        
        for token in special_tokens:
            text = text.replace(token, '')
            
        return text
        
    def extract_thinking(self, text: str) -> Dict[str, List[str]]:
        """
        Extract thinking portions instead of removing them
        Useful for debugging or analyzing model reasoning
        """
        thinking_segments = {}
        
        for pattern, _ in self.thinking_patterns:
            try:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    pattern_name = pattern.split('(')[0] if '(' in pattern else pattern
                    thinking_segments[pattern_name] = [
                        match[0] if isinstance(match, tuple) else match 
                        for match in matches
                    ]
            except Exception as e:
                logger.debug(f"Could not extract pattern {pattern}: {e}")
                
        return thinking_segments


class ChunkedResponseFilter:
    """Specialized filter for handling chunked responses"""
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        self.filter = ResponseFilter(model_config)
        self.chunk_history: List[str] = []
        self.continuation_patterns = [
            r"^\.\.\.\s*continuing",
            r"^continuing from",
            r"^\[continued\]",
        ]
        
    def filter_chunk(self, chunk_text: str, chunk_index: int, 
                     is_first: bool = False, is_last: bool = False) -> str:
        """
        Filter a single chunk with awareness of its position
        
        Args:
            chunk_text: Text chunk to filter
            chunk_index: Index of this chunk
            is_first: Whether this is the first chunk
            is_last: Whether this is the last chunk
            
        Returns:
            Filtered chunk text
        """
        # Apply standard filtering
        result = self.filter.filter(chunk_text)
        filtered_text = result.filtered_text
        
        # Handle continuation markers
        if not is_first:
            for pattern in self.continuation_patterns:
                filtered_text = re.sub(pattern, '', filtered_text, 
                                      flags=re.IGNORECASE)
                
        # Store for context
        self.chunk_history.append(filtered_text)
        
        # Additional processing for last chunk
        if is_last:
            # Remove any trailing incomplete sentences
            if filtered_text and not filtered_text.rstrip().endswith(('.', '!', '?')):
                # Try to find the last complete sentence
                last_sentence_end = max(
                    filtered_text.rfind('.'),
                    filtered_text.rfind('!'),
                    filtered_text.rfind('?')
                )
                if last_sentence_end > 0:
                    filtered_text = filtered_text[:last_sentence_end + 1]
                    
        return filtered_text
        
    def merge_chunks(self, chunks: List[str]) -> str:
        """
        Merge filtered chunks into coherent text
        
        Args:
            chunks: List of filtered chunk texts
            
        Returns:
            Merged text with smooth transitions
        """
        if not chunks:
            return ""
            
        # First pass: basic merge
        merged = '\n\n'.join(chunk.strip() for chunk in chunks if chunk.strip())
        
        # Second pass: smooth transitions
        # Remove duplicate content at chunk boundaries
        merged = self._remove_boundary_duplicates(merged)
        
        # Third pass: ensure consistent formatting
        merged = self.filter._preserve_structure(merged)
        
        return merged
        
    def _remove_boundary_duplicates(self, text: str) -> str:
        """Remove duplicate content that might appear at chunk boundaries"""
        lines = text.split('\n')
        cleaned_lines = []
        previous_line = ""
        
        for line in lines:
            # Check for duplicate or near-duplicate lines
            if line.strip() and previous_line.strip():
                # Simple similarity check (could be enhanced)
                if self._line_similarity(line, previous_line) < 0.8:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
                
            previous_line = line
            
        return '\n'.join(cleaned_lines)
        
    def _line_similarity(self, line1: str, line2: str) -> float:
        """Calculate simple similarity between two lines"""
        line1_words = set(line1.lower().split())
        line2_words = set(line2.lower().split())
        
        if not line1_words or not line2_words:
            return 0.0
            
        intersection = line1_words.intersection(line2_words)
        union = line1_words.union(line2_words)
        
        return len(intersection) / len(union) if union else 0.0


class DynamicTokenLimitCalculator:
    """Calculate optimal token limits based on content and model constraints"""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.base_prompt_tokens = 500  # Estimated tokens for system prompt
        
    def calculate_max_new_tokens(self, input_text: str, 
                                 target_ratio: float = 2.0) -> int:
        """
        Calculate optimal max_new_tokens based on input
        
        Args:
            input_text: Input text to process
            target_ratio: Desired output/input ratio
            
        Returns:
            Optimal max_new_tokens value
        """
        # Estimate input tokens (rough approximation)
        estimated_input_tokens = len(input_text) // 4  # ~4 chars per token
        
        # Add base prompt tokens
        total_input_tokens = estimated_input_tokens + self.base_prompt_tokens
        
        # Calculate available tokens
        available_tokens = self.model_config.max_context - total_input_tokens
        
        # Apply target ratio
        target_output_tokens = int(estimated_input_tokens * target_ratio)
        
        # Use the minimum of target and available
        max_new_tokens = min(target_output_tokens, available_tokens)
        
        # Apply bounds
        max_new_tokens = max(256, min(max_new_tokens, 8192))
        
        logger.debug(
            f"Calculated max_new_tokens: {max_new_tokens} "
            f"(input_tokens≈{estimated_input_tokens}, "
            f"available={available_tokens})"
        )
        
        return max_new_tokens
