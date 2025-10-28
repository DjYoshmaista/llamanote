# llamanote/formatting/base_formatter.py
"""
Base Formatter Module
Provides the base class and factory for text formatting.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from ..utils.logger import get_logger_conf
from ..config.settings import MARKDOWN_STYLES, MarkdownStyle
from ..core.types import ProcessingMode

logger = get_logger_conf(__name__)

# --- Reusable Pattern Matching Utilities (Refactoring Item 6) ---

class PatternMatcher:
    """Utility class for pattern matching logic."""
    
    @staticmethod
    def detect_from_patterns(text: str, patterns_map: Dict[Enum, List[str]]) -> Optional[Enum]:
        """
        Detects the most likely category based on keyword/regex patterns.
        """
        text_lower = text.lower()
        scores = {key: 0 for key in patterns_map}
        
        for key, patterns in patterns_map.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    scores[key] += 1
                    
        max_score = max(scores.values())
        if max_score > 0:
            # Return the key with the highest score
            # Note: This doesn't handle ties, just returns the first max
            return max(scores, key=scores.get)
        return None

class PatternReplacer:
    """Utility class for applying regex-based replacements."""
    
    def __init__(self, patterns: List[Tuple[str, str]]):
        self.compiled_patterns = []
        for pattern, replacement in patterns:
            try:
                self.compiled_patterns.append((re.compile(pattern, re.IGNORECASE | re.MULTILINE), replacement))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}. Skipping.")
    
    def replace(self, text: str) -> str:
        """Applies all compiled replacement patterns to the text."""
        for pattern, replacement in self.compiled_patterns:
            text = pattern.sub(replacement, text)
        return text

# --- Base Formatter ---

class BaseFormatter:
    """Base class for formatting processed text."""
    
    def __init__(self, style_name: str = "default"):
        self.style_name = style_name
        self.style_config = MARKDOWN_STYLES.get(style_name)
        if not self.style_config:
            logger.warning(f"Unknown style '{style_name}', using 'default' style.")
            self.style_config = MARKDOWN_STYLES["default"]
        logger.info(f"Initialized formatter with style: '{self.style_name}'")

    def format(self, text: str, **kwargs) -> str:
        """
        Main formatting method.
        
        Args:
            text: The text to format (assumed to be pre-filtered).
            **kwargs: Style-specific options (e.g., add_emotions).

        Returns:
            Formatted string.
        """
        # 1. Preserve code blocks
        text, code_blocks = self._extract_code_blocks(text)
        
        # 2. Apply basic formatting (implementation-specific)
        formatted_text = self._apply_format(text, **kwargs)
        
        # 3. Restore code blocks
        formatted_text = self._restore_code_blocks(formatted_text, code_blocks)
        
        # 4. Add structure (e.g., title)
        if kwargs.get('add_structure', True):
            formatted_text = self._add_document_structure(formatted_text)
            
        return formatted_text.strip()

    def _apply_format(self, text: str, **kwargs) -> str:
        """
        Subclasses override this to implement specific formatting logic.
        This base version just joins paragraphs.
        """
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return '\n\n'.join(paragraphs)

    def _add_document_structure(self, text: str) -> str:
        """Add overall document structure, like a title."""
        if not text.startswith('#') and "section" in self.style_config.structure_markers:
            title_template = self.style_config.structure_markers["section"]
            title = title_template.format("Processed Content")
            text = f"{title}\n{text}"
        return text

    def _extract_code_blocks(self, text: str) -> Tuple[str, List[str]]:
        """Extract code blocks to preserve them during formatting."""
        code_blocks = []
        # Pattern to find ```language\n...``` or ```\n...```
        pattern = re.compile(r"^(```[a-zA-Z]*\n[\s\S]*?\n```)", re.MULTILINE)
        
        def replacer(match):
            placeholder = f"__CODE_BLOCK__{len(code_blocks)}__"
            code_blocks.append(match.group(1))
            return placeholder
            
        text = pattern.sub(replacer, text)
        return text, code_blocks

    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """Restore preserved code blocks."""
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK__{i}__"
            text = text.replace(placeholder, block)
        return text


# --- Formatter Factory ---

def get_formatter(mode: ProcessingMode, style_name: Optional[str] = None) -> BaseFormatter:
    """
    Factory function to get the appropriate formatter based on mode or style.
    
    Args:
        mode: The processing mode (e.g., PODCAST, TECHNICAL).
        style_name: An optional specific style name (overrides mode default).

    Returns:
        An instance of a BaseFormatter subclass.
    """
    
    # Lazy load specific formatters to avoid circular imports if they grow
    from .podcast_formatter import PodcastFormatter
    from .technical_formatter import TechnicalFormatter
    from .narrative_formatter import NarrativeFormatter
    
    if style_name:
        # If a specific style is requested, try to find it
        if style_name == "podcast": return PodcastFormatter()
        if style_name == "technical": return TechnicalFormatter()
        if style_name == "narrative": return NarrativeFormatter()
        # Fallback to base formatter with that style config
        logger.debug(f"Using BaseFormatter with custom style '{style_name}'")
        return BaseFormatter(style_name)

    # If no style is specified, use the mode
    if mode == ProcessingMode.PODCAST:
        return PodcastFormatter()
    elif mode == ProcessingMode.TECHNICAL:
        return TechnicalFormatter()
    elif mode == ProcessingMode.NARRATIVE:
        return NarrativeFormatter()
    else:
        # Default for SUMMARY, CUSTOM, or others
        return BaseFormatter(style_name="default")
