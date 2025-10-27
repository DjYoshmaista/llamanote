"""
Markdown Formatter Module
Provides advanced markdown formatting with emotional markers and style options
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from logging_config import get_logger
from config import MARKDOWN_STYLES, MarkdownStyle

logger = get_logger(__name__)


class EmotionType(Enum):
    """Enumeration of emotion types for content marking"""
    NEUTRAL = "neutral"
    EXCITED = "excited"
    THOUGHTFUL = "thoughtful"
    SERIOUS = "serious"
    HUMOROUS = "humorous"
    SURPRISED = "surprised"
    QUESTIONING = "questioning"
    EMPHATIC = "emphatic"
    CAUTIOUS = "cautious"
    CONFIDENT = "confident"


class EmphasisLevel(Enum):
    """Levels of emphasis for text formatting"""
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    STRONG = 3
    MAXIMUM = 4


@dataclass
class FormattedSegment:
    """Represents a formatted text segment with metadata"""
    text: str
    emotion: Optional[EmotionType] = None
    emphasis: EmphasisLevel = EmphasisLevel.NONE
    speaker: Optional[str] = None
    is_quote: bool = False
    is_code: bool = False
    metadata: Dict[str, Any] = None


class MarkdownFormatter:
    """Advanced markdown formatter with emotional and structural markers"""
    
    def __init__(self, style: str = "podcast"):
        """Initialize formatter with specified style"""
        if style not in MARKDOWN_STYLES:
            logger.warning(f"Unknown style '{style}', using 'podcast' as default")
            style = "podcast"
            
        self.style_config = MARKDOWN_STYLES[style]
        self.style_name = style
        logger.info(f"Initialized MarkdownFormatter with '{style}' style")
        
    def format_text(self, 
                   text: str,
                   detect_emotions: bool = True,
                   add_structure: bool = True,
                   preserve_code: bool = True) -> str:
        """
        Format text with markdown and emotional markers
        
        Args:
            text: Input text to format
            detect_emotions: Whether to detect and mark emotions
            add_structure: Whether to add structural markers
            preserve_code: Whether to preserve code blocks
            
        Returns:
            Formatted markdown text
        """
        logger.debug(f"Formatting text of length {len(text)}")
        
        # Preserve code blocks if requested
        code_blocks = []
        if preserve_code:
            text, code_blocks = self._extract_code_blocks(text)
            
        # Split into segments for processing
        segments = self._segment_text(text)
        
        # Process each segment
        formatted_segments = []
        for segment in segments:
            # Detect emotion if enabled
            if detect_emotions:
                emotion = self._detect_emotion(segment)
            else:
                emotion = EmotionType.NEUTRAL
                
            # Detect emphasis level
            emphasis = self._detect_emphasis(segment)
            
            # Create formatted segment
            formatted = self._format_segment(segment, emotion, emphasis)
            formatted_segments.append(formatted)
            
        # Join segments
        formatted_text = self._join_segments(formatted_segments, add_structure)
        
        # Restore code blocks
        if code_blocks:
            formatted_text = self._restore_code_blocks(formatted_text, code_blocks)
            
        # Add document structure if requested
        if add_structure:
            formatted_text = self._add_document_structure(formatted_text)
            
        return formatted_text
        
    def _segment_text(self, text: str) -> List[str]:
        """Split text into processable segments"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        segments = []
        for para in paragraphs:
            # Further split long paragraphs by sentences
            if len(para) > 500:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                segments.extend(sentences)
            else:
                segments.append(para)
                
        return [s.strip() for s in segments if s.strip()]
        
    def _detect_emotion(self, text: str) -> EmotionType:
        """Detect emotion in text segment using keyword and pattern matching"""
        text_lower = text.lower()
        
        # Emotion detection patterns
        emotion_patterns = {
            EmotionType.EXCITED: [
                r'!+', r'amazing', r'incredible', r'fantastic', r'wonderful',
                r'awesome', r'excellent', r'brilliant', r'exciting'
            ],
            EmotionType.THOUGHTFUL: [
                r'perhaps', r'maybe', r'consider', r'think about', r'reflect',
                r'ponder', r'wonder', r'imagine', r'suppose'
            ],
            EmotionType.SERIOUS: [
                r'important', r'critical', r'essential', r'must', r'necessary',
                r'vital', r'crucial', r'fundamental', r'significant'
            ],
            EmotionType.HUMOROUS: [
                r'haha', r'lol', r'funny', r'joke', r'laugh', r'humor',
                r'amusing', r'hilarious', r'witty'
            ],
            EmotionType.SURPRISED: [
                r'surprisingly', r'unexpectedly', r'amazingly', r'shockingly',
                r'astonishing', r'remarkable', r'wow', r'oh my'
            ],
            EmotionType.QUESTIONING: [
                r'\?+', r'why', r'how', r'what if', r'could it be',
                r'wonder if', r'might be', r'possibly'
            ],
            EmotionType.CAUTIOUS: [
                r'careful', r'warning', r'caution', r'beware', r'note that',
                r'keep in mind', r'remember', r'don\'t forget'
            ],
            EmotionType.CONFIDENT: [
                r'definitely', r'certainly', r'absolutely', r'clearly',
                r'obviously', r'undoubtedly', r'surely', r'without doubt'
            ]
        }
        
        # Count matches for each emotion
        emotion_scores = {}
        for emotion, patterns in emotion_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            emotion_scores[emotion] = score
            
        # Return emotion with highest score, or NEUTRAL if no matches
        if max(emotion_scores.values()) > 0:
            return max(emotion_scores, key=emotion_scores.get)
        return EmotionType.NEUTRAL
        
    def _detect_emphasis(self, text: str) -> EmphasisLevel:
        """Detect emphasis level in text"""
        # Check for various emphasis indicators
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        exclamation_count = text.count('!')
        
        if caps_ratio > 0.7 or exclamation_count >= 3:
            return EmphasisLevel.MAXIMUM
        elif caps_ratio > 0.5 or exclamation_count >= 2:
            return EmphasisLevel.STRONG
        elif caps_ratio > 0.3 or exclamation_count >= 1:
            return EmphasisLevel.MEDIUM
        elif any(word.isupper() for word in text.split() if len(word) > 2):
            return EmphasisLevel.LIGHT
        else:
            return EmphasisLevel.NONE
            
    def _format_segment(self, text: str, emotion: EmotionType, 
                       emphasis: EmphasisLevel) -> str:
        """Format a text segment with emotion and emphasis markers"""
        formatted = text
        
        # Apply emphasis formatting
        if emphasis != EmphasisLevel.NONE:
            formatted = self._apply_emphasis(formatted, emphasis)
            
        # Apply emotion markers if available in style
        if emotion != EmotionType.NEUTRAL and self.style_config.emotion_markers:
            formatted = self._apply_emotion_marker(formatted, emotion)
            
        return formatted
        
    def _apply_emphasis(self, text: str, level: EmphasisLevel) -> str:
        """Apply emphasis formatting based on level"""
        markers = self.style_config.emphasis_markers
        
        if level == EmphasisLevel.LIGHT and "emphasis" in markers:
            # Emphasize key words (first and last significant words)
            words = text.split()
            if len(words) > 3:
                words[0] = markers["emphasis"].format(words[0])
                words[-1] = markers["emphasis"].format(words[-1])
                text = ' '.join(words)
                
        elif level == EmphasisLevel.MEDIUM and "emphasis" in markers:
            text = markers["emphasis"].format(text)
            
        elif level == EmphasisLevel.STRONG and "strong" in markers:
            text = markers["strong"].format(text)
            
        elif level == EmphasisLevel.MAXIMUM and "strong" in markers:
            # Use triple asterisks or combined markers
            text = f"***{text}***"
            
        return text
        
    def _apply_emotion_marker(self, text: str, emotion: EmotionType) -> str:
        """Apply emotion markers to text"""
        emotion_key = emotion.value
        
        if emotion_key in self.style_config.emotion_markers:
            marker_template = self.style_config.emotion_markers[emotion_key]
            return marker_template.format(text)
            
        return text
        
    def _join_segments(self, segments: List[str], add_structure: bool) -> str:
        """Join formatted segments with appropriate spacing"""
        if not add_structure:
            return '\n\n'.join(segments)
            
        # Group segments into logical sections
        output = []
        current_section = []
        
        for i, segment in enumerate(segments):
            current_section.append(segment)
            
            # Add section break every 3-5 segments for readability
            if len(current_section) >= 4 or i == len(segments) - 1:
                output.append('\n\n'.join(current_section))
                if i < len(segments) - 1:
                    # Add transition marker
                    if "transition" in self.style_config.structure_markers:
                        output.append(self.style_config.structure_markers["transition"])
                current_section = []
                
        return '\n'.join(output)
        
    def _add_document_structure(self, text: str) -> str:
        """Add overall document structure"""
        # This could be enhanced to detect natural section breaks
        # For now, just add a title if not present
        if not text.startswith('#'):
            title = self.style_config.structure_markers.get("section", "## {}").format("Processed Content")
            text = title + "\n\n" + text
            
        return text
        
    def _extract_code_blocks(self, text: str) -> Tuple[str, List[str]]:
        """Extract code blocks to preserve them during formatting"""
        code_blocks = []
        
        # Find all code blocks
        pattern = r'```[\s\S]*?```'
        matches = re.findall(pattern, text)
        
        for i, match in enumerate(matches):
            placeholder = f"__CODE_BLOCK_{i}__"
            text = text.replace(match, placeholder, 1)
            code_blocks.append(match)
            
        return text, code_blocks
        
    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """Restore preserved code blocks"""
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            text = text.replace(placeholder, block)
            
        return text


class PodcastFormatter(MarkdownFormatter):
    """Specialized formatter for podcast scripts"""
    
    def __init__(self):
        super().__init__(style="podcast")
        self.speaker_counter = 0
        self.current_speaker = None
        
    def format_dialogue(self, 
                       text: str,
                       speakers: Optional[List[str]] = None,
                       auto_detect_speakers: bool = True) -> str:
        """
        Format text as dialogue for podcast script
        
        Args:
            text: Input text
            speakers: List of speaker names
            auto_detect_speakers: Whether to auto-detect speaker changes
            
        Returns:
            Formatted podcast script
        """
        if speakers is None:
            speakers = ["Host", "Guest"]
            
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
                
            # Detect speaker change
            if auto_detect_speakers and self._is_speaker_change(line):
                self.current_speaker = self._get_next_speaker(speakers)
                formatted_lines.append(
                    self._format_speaker_line(self.current_speaker)
                )
                
            # Format the line with appropriate markers
            formatted_line = self._format_dialogue_line(line)
            formatted_lines.append(formatted_line)
            
        return '\n'.join(formatted_lines)
        
    def _is_speaker_change(self, line: str) -> bool:
        """Detect if this line indicates a speaker change"""
        # Simple heuristic: questions often indicate speaker change
        # This could be made more sophisticated
        indicators = [
            line.endswith('?'),
            line.startswith(('Well,', 'So,', 'Now,', 'Actually,')),
            len(line) < 20  # Short lines often indicate transitions
        ]
        return any(indicators)
        
    def _get_next_speaker(self, speakers: List[str]) -> str:
        """Get the next speaker in rotation"""
        self.speaker_counter = (self.speaker_counter + 1) % len(speakers)
        return speakers[self.speaker_counter]
        
    def _format_speaker_line(self, speaker: str) -> str:
        """Format a speaker introduction line"""
        if "speaker_change" in self.style_config.structure_markers:
            return self.style_config.structure_markers["speaker_change"].format(speaker)
        return f"\n**{speaker}:**\n"
        
    def _format_dialogue_line(self, line: str) -> str:
        """Format a line of dialogue with appropriate emotion and pacing markers"""
        # Detect pacing
        if '...' in line:
            # Slow, thoughtful delivery
            if "pause" in self.style_config.emphasis_markers:
                line = re.sub(r'\.\.\.', '...', line)  # Normalize ellipsis
                parts = line.split('...')
                line = self.style_config.emphasis_markers["pause"].format('...').join(parts)
                
        # Detect questions for rising intonation
        if line.endswith('?'):
            if "questioning" in self.style_config.emotion_markers:
                line = self.style_config.emotion_markers["questioning"].format(line)
                
        return line


class TechnicalFormatter(MarkdownFormatter):
    """Specialized formatter for technical documentation"""
    
    def __init__(self):
        super().__init__(style="technical")
        
    def format_with_headers(self, text: str, auto_generate_toc: bool = True) -> str:
        """
        Format technical text with proper headers and optionally generate TOC
        
        Args:
            text: Input text
            auto_generate_toc: Whether to generate table of contents
            
        Returns:
            Formatted technical document
        """
        # Split into sections
        sections = self._identify_sections(text)
        
        formatted_sections = []
        toc_entries = []
        
        for i, (title, content) in enumerate(sections):
            # Format section title
            if title:
                formatted_title = f"## {title}"
                formatted_sections.append(formatted_title)
                toc_entries.append(f"{i+1}. [{title}](#{title.lower().replace(' ', '-')})")
                
            # Format section content
            formatted_content = self._format_technical_content(content)
            formatted_sections.append(formatted_content)
            
        # Combine sections
        formatted_text = '\n\n'.join(formatted_sections)
        
        # Add TOC if requested
        if auto_generate_toc and toc_entries:
            toc = "## Table of Contents\n\n" + '\n'.join(toc_entries)
            formatted_text = toc + '\n\n' + formatted_text
            
        return formatted_text
        
    def _identify_sections(self, text: str) -> List[Tuple[str, str]]:
        """Identify logical sections in text"""
        # Simple implementation - could be enhanced with NLP
        paragraphs = text.split('\n\n')
        
        sections = []
        current_section = []
        current_title = "Introduction"
        
        for para in paragraphs:
            # Check if paragraph seems like a section header
            if len(para) < 100 and not para.endswith('.'):
                # Likely a header
                if current_section:
                    sections.append((current_title, '\n\n'.join(current_section)))
                current_title = para.strip()
                current_section = []
            else:
                current_section.append(para)
                
        # Add final section
        if current_section:
            sections.append((current_title, '\n\n'.join(current_section)))
            
        return sections
        
    def _format_technical_content(self, content: str) -> str:
        """Format technical content with appropriate markers"""
        # Detect and format code snippets
        content = self._format_inline_code(content)
        
        # Detect and format important notes
        content = self._format_notes(content)
        
        # Format lists
        content = self._format_lists(content)
        
        return content
        
    def _format_inline_code(self, text: str) -> str:
        """Format inline code snippets"""
        # Detect potential code terms
        code_pattern = r'\b([A-Z][a-zA-Z0-9_]*|[a-z]+_[a-z]+|[a-z]+\(\))\b'
        
        def replace_code(match):
            term = match.group(1)
            # Don't format if already in backticks
            if not (match.start() > 0 and text[match.start()-1] == '`'):
                return f"`{term}`"
            return term
            
        return re.sub(code_pattern, replace_code, text)
        
    def _format_notes(self, text: str) -> str:
        """Format important notes and warnings"""
        note_patterns = [
            (r'(Note|NOTE):\s*(.+)', '📝 **Note:** \\2'),
            (r'(Warning|WARNING):\s*(.+)', '⚠️ **Warning:** \\2'),
            (r'(Tip|TIP):\s*(.+)', '💡 **Tip:** \\2'),
            (r'(Important|IMPORTANT):\s*(.+)', '❗ **Important:** \\2'),
        ]
        
        for pattern, replacement in note_patterns:
            text = re.sub(pattern, replacement, text)
            
        return text
        
    def _format_lists(self, text: str) -> str:
        """Format bullet points and numbered lists"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Detect list items
            if re.match(r'^\d+[\.\)]\s', line):
                # Numbered list - already formatted
                formatted_lines.append(line)
            elif re.match(r'^[-*•]\s', line):
                # Bullet list - normalize to markdown
                formatted_lines.append(re.sub(r'^[-*•]\s', '- ', line))
            elif line.strip() and len(line) < 100 and ':' in line:
                # Possible definition list
                parts = line.split(':', 1)
                if len(parts) == 2:
                    formatted_lines.append(f"**{parts[0].strip()}:** {parts[1].strip()}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
                
        return '\n'.join(formatted_lines)
