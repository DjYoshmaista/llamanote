# llamanote/formatting/podcast_formatter.py
"""
Podcast Formatter Module
Specialized formatter for creating podcast scripts from text.
"""

import re
from typing import List, Optional

from .base_formatter import BaseFormatter, PatternMatcher
from ..core.types import EmotionType # Assuming EmotionType is in core.types
from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

# --- Emotion/Pacing Patterns for Podcast (Refactoring Item 6) ---

class PodcastEmotionPatterns(Enum):
    EXCITED = [
        r'!', r'amazing', r'incredible', r'fantastic', r'wonderful',
        r'awesome', r'excellent', r'brilliant', r'exciting', r'love this'
    ]
    THOUGHTFUL = [
        r'\.\.\.', r'hmm', r'I wonder', r'perhaps', r'maybe', 'consider',
        'think about', 'reflect', 'ponder', 'imagine', 'suppose'
    ]
    SERIOUS = [
        r'important', r'critical', r'essential', r'must', r'necessary',
        r'vital', r'crucial', r'fundamental', r'significant', r'serious'
    ]
    HUMOROUS = [
        r'haha', r'lol', r'funny', r'joke', r'laugh', 'hilarious', 'witty'
    ]
    SURPRISED = [
        r'surprising(ly)?', r'unexpected(ly)?', r'astonishing', r'remarkable', r'wow', 'oh my'
    ]
    QUESTIONING = [
        r'\?', r'why', r'how', r'what if', r'really\?'
    ]

class PodcastEmphasisPatterns(Enum):
    STRONG = [
        r'\b[A-Z]{5,}\b', # All-caps word (5+ chars)
        r'!', # Exclamation marks
    ]
    EMPHASIS = [
        r'\b(very|really|so) [a-zA-Z]+' # Simple emphasis words
    ]


class PodcastFormatter(BaseFormatter):
    """
    Formats text into a structured podcast script, adding speaker
    labels, and optional emotional/pacing markers.
    """
    
    def __init__(self):
        super().__init__(style="podcast")
        self.speaker_counter = 0
        self.current_speaker = "Host" # Default start
        self.speaker_pattern = re.compile(r"^\s*\[?\s*(SPEAKER \w+|HOST|GUEST)\s*\]?:\s*", re.IGNORECASE)
        self.emotion_detector = PatternMatcher()

    def format(self, text: str, **kwargs) -> str:
        """
        Overrides base format method to apply dialogue-specific formatting.

        Args:
            text: The text to format.
            **kwargs:
                auto_detect_speakers (bool): If True, tries to assign "Host" and "Guest".
                add_emotions (bool): If True, adds emotion/emphasis markers.

        Returns:
            Formatted podcast script as a markdown string.
        """
        auto_detect = kwargs.get('auto_detect_speakers', True)
        add_emotions = kwargs.get('add_emotions', True)

        # Preserve code blocks first (unlikely in podcast, but respects base class)
        text, code_blocks = self._extract_code_blocks(text)
        
        # Split into paragraphs, which we treat as speaker turns
        paragraphs = re.split(r'\n\s*\n', text)
        
        formatted_segments = []
        self.speaker_counter = 0 # Reset for each call
        self.current_speaker = "Host"

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            speaker_name, content = self._extract_speaker(para, auto_detect)
            
            # Format the speaker tag
            if "speaker_change" in self.style_config.structure_markers:
                 speaker_line = self.style_config.structure_markers["speaker_change"].format(speaker_name)
                 formatted_segments.append(speaker_line)
            else: # Fallback
                 formatted_segments.append(f"\n**[{speaker_name}]:**\n")

            # Format the content
            formatted_content = self._format_dialogue_line(content, add_emotions)
            formatted_segments.append(formatted_content)
            
        # Join segments (don't add extra newlines, speaker tags already have them)
        formatted_text = ''.join(formatted_segments)
        
        # Restore code blocks (again, unlikely but safe)
        formatted_text = self._restore_code_blocks(formatted_text, code_blocks)
        
        # Add main title
        if kwargs.get('add_structure', True):
            formatted_text = self._add_document_structure(formatted_text)
            
        return formatted_text.strip()

    def _extract_speaker(self, paragraph: str, auto_detect: bool) -> Tuple[str, str]:
        """
        Determines the speaker for a paragraph.
        If a speaker tag exists, use it.
        If not and auto_detect is on, alternate.
        """
        match = self.speaker_pattern.match(paragraph)
        
        if match:
            # Explicit speaker tag found
            speaker_tag = match.group(1).upper()
            if "HOST" in speaker_tag:
                 self.current_speaker = "Host"
            elif "GUEST" in speaker_tag or "SPEAKER" in speaker_tag:
                 self.current_speaker = "Guest"
            # else: keep self.current_speaker as is? Or use the tag?
            # Let's use the standardized version.
            
            content = paragraph[match.end():].strip()
            return self.current_speaker, content
            
        elif auto_detect:
            # No tag found, assume it's the *next* speaker
            if self.current_speaker == "Host":
                self.current_speaker = "Guest"
            else:
                self.current_speaker = "Host"
            return self.current_speaker, paragraph
            
        else:
            # No auto-detect, assume same speaker
            return self.current_speaker, paragraph

    def _format_dialogue_line(self, line: str, add_emotions: bool) -> str:
        """Formats a single line of dialogue with emotion/emphasis."""
        
        # Apply standard text processing (e.g., whitespace normalization)
        line = re.sub(r'\s+', ' ', line).strip()

        if add_emotions:
            # Detect primary emotion for the line
            emotion = self.emotion_detector.detect_from_patterns(line, PodcastEmotionPatterns)
            
            # Apply emotion marker from style config
            if emotion and emotion.value in self.style_config.emotion_markers:
                marker_template = self.style_config.emotion_markers[emotion.value]
                line = marker_template.format(line)

            # Apply emphasis (this is a simple example)
            if self.emotion_detector.detect_from_patterns(line, {"strong": PodcastEmphasisPatterns.STRONG.value}):
                 if "strong" in self.style_config.emphasis_markers:
                      line = self.style_config.emphasis_markers["strong"].format(line)

        # Apply simple pause marker
        if "pause" in self.style_config.emphasis_markers:
             line = line.replace("...", self.style_config.emphasis_markers["pause"].format("..."))

        return line
