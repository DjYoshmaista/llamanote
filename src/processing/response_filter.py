# llamanote/processing/response_filter.py
"""
Response Filter Module
Filters out thinking tokens and other unwanted patterns from LLM responses.
"""

import re
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field

from ..utils.logger import get_logger_conf
from ..core.types import FilterResult
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

# Patterns for instruction regurgitation (meta-commentary about preprocessing)
INSTRUCTION_REGURGITATION_PATTERNS = [
    # Preprocessing instructions leaked into output
    (r"(?i)(?:you are|i am|the user is) a (world class )?text (pre-)?processor", ""),
    (r"(?i)(?:the )?raw data is riddled with", ""),
    (r"(?i)the goal is to parse and return", ""),
    (r"(?i)the response should (?:be )?(very )?smart and aggressive", ""),
    (r"(?i)(?:do not|don't) start with (?:any )?(?:markdown|summaries)", ""),
    (r"(?i)(?:remove|translate) (?:any )?details? (?:that )?(?:are |is )?(?:lost|useless|misunderstood)", ""),
    (r"(?i)the pre-?processor should (?:be )?(consistently|constantly)", ""),
    (r"(?i)keep returning the processed text", ""),
    (r"(?i)no acknowledgements or period at the end", ""),
    (r"(?i)clean\s*(?:up|this)\s*text\s*from\s*a\s*pdf", ""),
    # Repetitive meta-instructions
    (r"(?i)(?:this|the response) should start (?:directly )?with", ""),
    (r"(?i)(?:this is|it's) a rule (?:which|that) (?:is )?(?:constantly|consistently)", ""),
    # Podcast-specific instruction leaks
    (r"(?i)you will receive an? (?:outline|OUTLINE)", ""),
    (r"(?i)(?:the )?(?:outline|OUTLINE) (?:showing|shows) topics? to cover", ""),
    (r"(?i)(?:and |with )?(?:content|CONTENT) with (?:the )?source material", ""),
    (r"(?i)follow (?:the )?outline structure", ""),
    (r"(?i)using? information from (?:the )?content", ""),
    (r"(?i)(?:create|write) natural,? engaging dialogue", ""),
    (r"(?i)you are a podcast (?:script writer|producer)", ""),
    (r"(?i)raw text follows:?\s*$", ""),
]

# Tokens that are often artifacts and should just be removed
ARTIFACT_TOKENS = [
    '<|endoftext|>', '<|end|>', '<|assistant|>', '<|system|>', '<|user|>',
    '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>', '<s>', '</s>',
    'ASSISTANT:', 'USER:', 'SYSTEM:',
    '<|',  # Partial tokens from your example
    '|:',
    '**<|',
]

# Heavy citation patterns that indicate reference sections leaking through
CITATION_EXPLOSION_PATTERNS = [
    # Remove author initials patterns (J. J. Hopfield, E. D. Schoedel, etc.)
    (r'\b[A-Z]\.\s+(?:[A-Z]\.\s+)+\w+', ''),
    # Remove conference/journal patterns
    (r'(?:Proc\.|Proceedings?|Conference)\s+(?:of|on)\s+(?:the\s+)?\d+(?:th|st|nd|rd)?', ''),
    # Remove "Accessed DATE" patterns
    (r'Accessed\s+\d+\s+\w+\s+\d{4}\.?', ''),
    # Remove lines that are mostly numbers, dots, and letters (citation format)
    (r'^\s*[\d\.\s]+[A-Z]\..*$', '', re.MULTILINE),
    # Remove Sci. Am., Nat. Commun., etc. journal abbreviations
    (r'\b(?:Sci\.|Nat\.|Proc\.|J\.)\s+(?:Am\.|Commun\.|Natl\.|Acad\.)\b', ''),
]

# Structural labels that leak from prompt formatting (remove these at start of lines)
STRUCTURAL_LABEL_PATTERNS = [
    (r"^OUTLINE:\s*$", "", re.MULTILINE),
    (r"^CONTENT:\s*$", "", re.MULTILINE),
    (r"^Content to outline:\s*$", "", re.MULTILINE),
    (r"^Text:\s*$", "", re.MULTILINE),
    (r"^Raw text follows:\s*$", "", re.MULTILINE),
]

class ResponseFilter:
    """
    Filters thinking tokens, acknowledgments, and other patterns from LLM responses.
    """
    
    def __init__(self, model_config: Optional[ModelEntry] = None):
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

        # Compile instruction regurgitation patterns
        self.instruction_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.MULTILINE), replacement)
            for pattern, replacement in INSTRUCTION_REGURGITATION_PATTERNS
        ]

        # Compile structural label patterns (OUTLINE:, CONTENT:, etc.)
        self.structural_label_patterns = [
            (re.compile(pattern, flags), replacement)
            for pattern, replacement, flags in STRUCTURAL_LABEL_PATTERNS
        ]

        # Compile citation explosion patterns
        self.citation_patterns = []
        for item in CITATION_EXPLOSION_PATTERNS:
            if len(item) == 3:
                pattern, replacement, flags = item
                self.citation_patterns.append((re.compile(pattern, flags), replacement))
            else:
                pattern, replacement = item
                self.citation_patterns.append((re.compile(pattern), replacement))

        logger.info(f"Initialized ResponseFilter with {len(self.patterns)} thinking patterns, {len(self.ack_patterns)} acknowledgment patterns, {len(self.instruction_patterns)} instruction cleanup patterns, {len(self.structural_label_patterns)} structural label patterns, and {len(self.citation_patterns)} citation cleanup patterns.")
        
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

        # 3. Remove instruction regurgitation (meta-commentary)
        if remove_acknowledgments:  # Use same flag since it's similar cleanup
            for pattern, replacement in self.instruction_patterns:
                matches = pattern.findall(filtered_text)
                if matches:
                    removed_segments.append(f"[instruction]: {str(matches[:2])[:75]}...")
                filtered_text, num_subs = pattern.subn(replacement, filtered_text)
                if num_subs > 0:
                    filter_stats.setdefault('instruction_segments', 0)
                    filter_stats['instruction_segments'] += num_subs

        # 4. Remove structural labels (OUTLINE:, CONTENT:, etc.)
        for pattern, replacement in self.structural_label_patterns:
            filtered_text, num_subs = pattern.subn(replacement, filtered_text)
            if num_subs > 0:
                filter_stats.setdefault('structural_labels_removed', 0)
                filter_stats['structural_labels_removed'] += num_subs

        # 5. Remove artifact tokens
        for pattern in self.artifact_patterns:
            filtered_text, num_subs = pattern.subn("", filtered_text)
            filter_stats['artifacts_removed'] += num_subs

        # 6. Remove citation explosion patterns (CRITICAL for podcast mode)
        for pattern, replacement in self.citation_patterns:
            filtered_text, num_subs = pattern.subn(replacement, filtered_text)
            if num_subs > 0:
                filter_stats.setdefault('citations_removed', 0)
                filter_stats['citations_removed'] += num_subs

        # 7. Detect and truncate repetition loops
        filtered_text = self._detect_repetition_loop(filtered_text)

        # 8. Remove lines with excessive symbol density
        filtered_text = self._remove_symbol_heavy_lines(filtered_text)

        # 9. Final whitespace cleanup
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

    def _detect_repetition_loop(self, text: str, window: int = 30) -> str:
        """
        Detect and truncate text at first repetition loop.

        Args:
            text: Input text
            window: Size of text window to check for repetition

        Returns:
            Text truncated before repetition loop
        """
        words = text.split()
        if len(words) < window * 2:
            return text

        for i in range(len(words) - window):
            chunk = ' '.join(words[i:i+window])
            # Check if this chunk appears again in the next 100 words
            search_range = ' '.join(words[i+window:min(i+window+100, len(words))])
            if chunk in search_range:
                logger.warning(f"Detected repetition loop at word {i}, truncating")
                return ' '.join(words[:i])
        return text

    def _remove_symbol_heavy_lines(self, text: str, threshold: float = 0.6) -> str:
        """
        Remove lines with excessive non-alphanumeric content.

        Args:
            text: Input text
            threshold: Minimum ratio of alphanumeric characters (0-1)

        Returns:
            Text with symbol-heavy lines removed
        """
        lines = text.split('\n')
        cleaned = []

        for line in lines:
            if len(line.strip()) < 5:
                cleaned.append(line)  # Keep short lines (spacing)
                continue

            # Calculate alphanumeric ratio
            alphanum_count = sum(1 for c in line if c.isalnum() or c.isspace())
            ratio = alphanum_count / len(line) if len(line) > 0 else 1.0

            if ratio >= threshold:
                cleaned.append(line)
            else:
                logger.debug(f"Removed symbol-heavy line (ratio={ratio:.2f}): {line[:60]}")

        return '\n'.join(cleaned)

class ChunkedResponseFilter:
    """
    Specialized filter for handling chunked responses, focusing on stitching.
    """
    
    def __init__(self, model_config: Optional[ModelEntry] = None):
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
                remove_acknowledgments=True # Always remove acks
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
