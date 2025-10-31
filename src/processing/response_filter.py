# llamanote/processing/response_filter.py
"""
Response Filter Module
Filters out thinking tokens and other unwanted patterns from LLM responses.
"""
for __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Any, Set, TYPE_CHECKING
from dataclasses import dataclass, field

from ..utils.logger import get_logger_conf
from ..core.types import FilterResult

if TYPE_CHECKING:
    from ..models.registry import ModelEntry

logger = get_logger_conf(__name__)

# Default patterns to remove. These can be extended by model-specific patterns.
DEFAULT_THINKING_PATTERNS = [
    (r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    (r"<\|thinking\|>(.*?)<\|/thinking\|>", re.DOTALL | re.IGNORECASE),
    (r"\[THINK\](.*?)\[/THINK\]", re.DOTALL | re.IGNORECASE),
    (r"```thinking(.*?)```", re.DOTALL | re.IGNORECASE),
    (r"\(thinking:.*?\)", re.IGNORECASE),
    (r"\[internal:.*?\]", re.IGNORECASE),
]

DEFAULT_ACKNOWLEDGMENT_PATTERNS = [
    (r"^\s*(Sure|Certainly|Of course|I understand|I'll|Let me)[\s,!.]*(\n|Here's?)", ""), # At start of response
    (r"^Here's? (the|your|a) (cleaned|processed|formatted) text( below)?:?\s*\n+", ""), # At start of response
    (r"^I've (processed|cleaned|formatted) the text for you:\s*\n+", ""), # At start of response
]

# Tokens that are often artifacts and should just be removed
ARTIFACT_TOKENS = [
    '<|endoftext|>', '<|end|>', '<|assistant|>', '<|system|>', '<|user|>',
    '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>', '<s>', '</s>',
    'ASSISTANT:', 'USER:', 'SYSTEM:',
]

class ResponseFilter:
    """
    Filters thinking tokens, acknowledgments, and other patterns from LLM responses.
    """
    
    def __init__(self, model_config: Optional['ModelEntry'] = None):
        self.patterns: List[Tuple[re.Pattern, str]] = []
        self.artifact_patterns: List[re.Pattern] = []
        
        # Compile default thinking patterns
        for pattern, flags in DEFAULT_THINKING_PATTERNS:
            self.patterns.append((re.compile(pattern, flags), ""))
            
        # Compile artifact tokens for exact match removal
        for token in ARTIFACT_TOKENS:
            self.artifact_patterns.append(re.compile(re.escape(token)))

        # Add model-specific thinking tokens if provided
        if model_config and model_config.supports_thinking and model_config.thinking_tokens:
            for i in range(0, len(model_config.thinking_tokens), 2):
                if i + 1 < len(model_config.thinking_tokens):
                    start_tag = re.escape(model_config.thinking_tokens[i])
                    end_tag = re.escape(model_config.thinking_tokens[i+1])
                    # Add pattern: start_tag(.*?)end_tag
                    pattern = f"{start_tag}(.*?){end_tag}"
                    self.patterns.append((re.compile(pattern, re.DOTALL | re.IGNORECASE), ""))
            logger.info(f"Added {len(model_config.thinking_tokens)//2} model-specific thinking patterns.")

        # Compile acknowledgment patterns
        self.ack_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.MULTILINE), replacement)
            for pattern, replacement in DEFAULT_ACKNOWLEDGMENT_PATTERNS
        ]

        logger.info(f"Initialized ResponseFilter with {len(self.patterns)} thinking patterns and {len(self.ack_patterns)} acknowledgment patterns.")
        
    def filter(self, text: str, 
              remove_thinking: bool = True,
              remove_acknowledgments: bool = True) -> FilterResult:
        """
        Filter unwanted content from text.

        Args:
            text: Input text to filter
            remove_thinking: Remove configured thinking patterns
            remove_acknowledgments: Remove common AI acknowledgments
            
        Returns:
            FilterResult object containing filtered text and statistics
        """
        if not text:
            return FilterResult(original_text="", filtered_text="", removed_segments=[], filter_stats={})

        original_text = text
        filtered_text = text
        removed_segments = []
        filter_stats = {"thinking_segments": 0, "acknowledgment_segments": 0, "artifacts_removed": 0}
        
        # 1. Remove thinking patterns
        if remove_thinking:
            for pattern, replacement in self.patterns:
                matches = pattern.finditer(filtered_text)
                count = 0
                for match in matches:
                    segment = match.group(1) if match.groups() else match.group(0)
                    if segment and len(segment.strip()) > 5: # Only log non-trivial removals
                        removed_segments.append(f"[thinking]: {segment[:75].strip()}...")
                        count += 1
                
                if count > 0:
                    filter_stats['thinking_segments'] += count
                    filtered_text = pattern.sub(replacement, filtered_text)

        # 2. Remove acknowledgment patterns (only at the beginning)
        if remove_acknowledgments:
            for pattern, replacement in self.ack_patterns:
                 # Apply sub() only once at the start of the string
                 filtered_text, num_subs = pattern.subn(replacement, filtered_text, count=1)
                 if num_subs > 0:
                     filter_stats['acknowledgment_segments'] += num_subs

        # 3. Remove artifact tokens
        for pattern in self.artifact_patterns:
            filtered_text, num_subs = pattern.subn("", filtered_text)
            filter_stats['artifacts_removed'] += num_subs
            
        # 4. Final whitespace cleanup
        # Remove leading/trailing whitespace
        filtered_text = filtered_text.strip()
        # Consolidate multiple blank lines into one
        filtered_text = re.sub(r'\n(\s*\n)+', '\n\n', filtered_text)

        # Log statistics
        removed_chars = len(original_text) - len(filtered_text)
        removal_ratio = (removed_chars / len(original_text)) if original_text else 0.0
        
        logger.debug(
            f"Filtering complete: Removed {removed_chars} chars ({removal_ratio:.1%}). "
            f"Stats: {filter_stats}"
        )
        
        return FilterResult(
            original_text=original_text,
            filtered_text=filtered_text,
            removed_segments=removed_segments,
            filter_stats=filter_stats
        )

class ChunkedResponseFilter:
    """
    Specialized filter for handling chunked responses, focusing on stitching.
    """
    
    def __init__(self, model_config: Optional['ModelEntry'] = None):
        self.filter = ResponseFilter(model_config)
        self.continuation_patterns = [
            re.compile(r"^\.\.\.\s*continuing.*(?:\n|$)", re.IGNORECASE),
            re.compile(r"^continuing from.*(?:\n|$)", re.IGNORECASE),
            re.compile(r"^\[continued\]\s*", re.IGNORECASE),
        ]
        
    def filter_and_merge(self, 
                         chunks: List[str], 
                         remove_thinking: bool = True, 
                         remove_acknowledgments: bool = True) -> str:
        """
        Filters a list of text chunks and merges them intelligently.

        Args:
            chunks: List of raw text chunks from the LLM.
            remove_thinking: Whether to remove thinking tags.
            remove_acknowledgments: Whether to remove acknowledgments.

        Returns:
            A single, cleaned, and merged string.
        """
        if not chunks:
            return ""
            
        filtered_chunks = []
        
        # 1. Filter each chunk individually
        for i, chunk_text in enumerate(chunks):
            # Apply standard filtering
            result = self.filter.filter(
                chunk_text,
                remove_thinking=remove_thinking,
                remove_acknowledgments=(i == 0) # Only remove acks from the first chunk
            )
            
            filtered_text = result.filtered_text
            
            # 2. Handle continuation markers
            if i > 0: # Don't remove from the first chunk
                for pattern in self.continuation_patterns:
                    filtered_text = pattern.sub('', filtered_text)
            
            # 3. Handle artifacts from chunking (e.g., incomplete last sentence)
            # This logic is imperfect and best-effort.
            if i < len(chunks) - 1: # Not the last chunk
                # Find last sentence-ending punctuation
                last_punc = max(filtered_text.rfind(p) for p in ".!?")
                if last_punc != -1:
                    # Check if there's significant text after the last punctuation
                    trailing_text = filtered_text[last_punc+1:].strip()
                    if len(trailing_text) < 20 and not re.search(r'[a-zA-Z]', trailing_text):
                         # Likely just artifacts or incomplete fragments, trim it
                         filtered_text = filtered_text[:last_punc+1]
                # else: no punctuation, keep the whole chunk
            
            filtered_chunks.append(filtered_text.strip())

        # 4. Merge chunks
        # Join with double newline to respect paragraph breaks
        merged_text = '\n\n'.join(chunk for chunk in filtered_chunks if chunk)
        
        # 5. Final pass to normalize whitespace again
        merged_text = re.sub(r'\n{3,}', '\n\n', merged_text).strip()
        
        return merged_text
