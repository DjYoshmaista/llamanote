# llamanote/formatting/narrative_formatter.py
"""
Narrative Formatter Module
Specialized formatter for narrative text, focusing on dialogue and scene breaks.
"""

import re
from typing import List

from .base_formatter import BaseFormatter
from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

class NarrativeFormatter(BaseFormatter):
    """
    Formats text into a narrative or story-like structure,
    identifying and formatting dialogue and scene breaks.
    """
    
    def __init__(self):
        super().__init__(style="narrative")
        # Regex to identify lines that are likely dialogue
        # Matches lines starting with a quote or a hyphen/em-dash (common in fiction)
        self.dialogue_pattern = re.compile(r'^\s*["“\'—–-](.*?)["”\']?\s*$', re.MULTILINE)
        # Regex for scene breaks (e.g., ***, ---, #)
        self.scene_break_pattern = re.compile(r'^\s*([\*\-\#]\s*){3,}\s*$', re.MULTILINE)

    def _apply_format(self, text: str, **kwargs) -> str:
        """
        Overrides base format method to apply narrative formatting.

        Args:
            text: The text to format.
            **kwargs: Ignored for this formatter currently.

        Returns:
            Formatted narrative markdown string.
        """
        
        # 1. Standardize scene breaks
        if "break" in self.style_config.structure_markers:
            break_marker = self.style_config.structure_markers["break"]
            text = self.scene_break_pattern.sub(f"\n{break_marker}\n", text)
        
        # 2. Format dialogue
        # This is complex. A simple approach:
        # Find paragraphs that are *only* dialogue and format them.
        
        paragraphs = re.split(r'\n\s*\n', text)
        formatted_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if the entire paragraph is dialogue
            if self.dialogue_pattern.fullmatch(para):
                # Apply dialogue formatting
                if "dialogue" in self.style_config.emotion_markers: # Using 'emotion' markers for type
                    formatted_paragraphs.append(self.style_config.emotion_markers["dialogue"].format(para))
                else:
                    # Default: indent dialogue
                    formatted_paragraphs.append(f"> {para}")
            else:
                # Standard paragraph
                # Apply emphasis to parts of it
                para_with_emphasis = self._apply_emphasis(para)
                formatted_paragraphs.append(para_with_emphasis)
                
        formatted_text = '\n\n'.join(formatted_paragraphs)
        
        return formatted_text
        
    def _apply_emphasis(self, paragraph: str) -> str:
        """
        Applies emphasis markers (italic, bold) to narrative text.
        This is a simple placeholder. Real implementation would be complex.
        """
        # Example: Emphasize text in *asterisks* (already markdown)
        # or _underscores_ (already markdown)
        
        # A more complex rule: Emphasize ALL CAPS words
        if "strong" in self.style_config.emphasis_markers:
             repl_template = self.style_config.emphasis_markers["strong"]
             # Find words with 5+ uppercase letters
             paragraph = re.sub(r'(\b[A-Z]{5,}\b)', lambda m: repl_template.format(m.group(1)), paragraph)
        
        return paragraph
