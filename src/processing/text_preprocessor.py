# llamanote/processing/text_preprocessor.py
"""
Text Preprocessing and Cleaning Module
Handles normalization, whitespace fixing, and audio-specific cleaning.
"""

import re
from typing import List, Dict

from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

class TextPreprocessor:
    """Cleans and preprocesses raw text for various outputs (LLM, audio)."""

    def __init__(self):
        self.logger = get_logger_conf(f"{__name__}.TextPreprocessor")

        # --- General Preprocessing Regex ---
        self._url_regex = re.compile(r'https?://\S+|www\.\S+')
        self._email_regex = re.compile(r'\S+@\S+\.\S+')
        self._multi_newline_regex = re.compile(r'\n{3,}')
        self._multi_space_regex = re.compile(r' {2,}')

        # --- PDF Artifact Regex ---
        self._pdf_hyphen_regex = re.compile(r'(\w+)-\n(\w+)')
        self._pdf_punct_space_regex = re.compile(r'\s+([.,;!?])')
        self._pdf_punct_word_regex = re.compile(r'([.,;!?])(\w)')
        self._pdf_page_num_regex = re.compile(r'\n\s*\d+\s*\n|\nPage \d+ of \d+\n', re.IGNORECASE)
        self._pdf_bullet_regex = re.compile(r'^[•·■□▪▫◦‣⁃]\s+', re.MULTILINE)

        # --- Reference Section Detection ---
        # Patterns to detect common reference section headers
        self._ref_section_patterns = [
            re.compile(r'\n\s*(?:REFERENCES|References|Bibliography|BIBLIOGRAPHY|Works Cited|WORKS CITED|Literature Cited)\s*\n', re.IGNORECASE),
            re.compile(r'\n\s*\[\d+\]\s+\w', re.MULTILINE),  # Numbered reference list
            # Detect numbered author citations (1. J. J. Hopfield, 2. E. D. Schoedel, etc.)
            re.compile(r'\n\d+\.\s+[A-Z]\.\s+(?:[A-Z]\.\s+)*\w+', re.MULTILINE),
            # Detect Acknowledgments section
            re.compile(r'\n\s*(?:Acknowledgments?|ACKNOWLEDGMENTS?|Funding|FUNDING)\s*\n', re.IGNORECASE),
            # Detect dense citation blocks (3+ consecutive lines starting with numbers)
            re.compile(r'(?:\n\d+\..*){3,}', re.MULTILINE),
        ]

        # --- Audio Cleaning Regex ---
        self._citation_regex = re.compile(r'\[\d+(, ?\d+)*\]') # [1], [1, 2]
        self._year_regex = re.compile(r'\(\d{4}\)') # (2024)
        self._latex_inline_regex = re.compile(r'\$.*?\$')
        self._latex_block_regex = re.compile(r'\\\[.*?\\\]|\\\(.*?\\\)', re.DOTALL) # \[...\] or \(...\)
        self._latex_cmd_regex = re.compile(r'\\[a-zA-Z]+\{.*?\}') # \cmd{...}

        # --- Character Normalization Maps ---
        self._encoding_map = {
            '\u201c': '"', # “
            '\u201d': '"', # ”
            '\u2018': "'", # ‘
            '\u2019': "'", # ’
            '\u2013': '-', # – (en-dash)
            '\u2014': '--',# — (em-dash)
            '\u2026': '...', # …
            '\u200b': '',  # Zero-width space
            '\ufeff': '',  # BOM
            '\xa0': ' ',   # Non-breaking space
            '\u00AD': '',  # Soft hyphen
            '\u200C': '',  # Zero Width Non-Joiner
            '\u200D': '',  # Zero Width Joiner
            '\u00A0': ' ',  # No-break space
            '\u00B7': '.',  # Middle dot
            '\u2022': '-',  # Bullet
            '\u25CF': '-',  # Black circle (another bullet)
            '\u25BA': '>',  # Black right-pointing pointer
            '\u25C4': '<',  # Black left-pointing pointer
        }
        
        self._audio_abbreviations = {
            'Dr.': 'Doctor', 'Mr.': 'Mister', 'Mrs.': 'Missus', 'Ms.': 'Miss',
            'Prof.': 'Professor', 'Sr.': 'Senior', 'Jr.': 'Junior',
            'Ph.D.': 'PhD', 'M.D.': 'MD',
            'B.S.': 'Bachelor of Science', 'M.S.': 'Master of Science',
            'i.e.': 'that is', 'e.g.': 'for example', 'etc.': 'et cetera',
            'vs.': 'versus', 'Inc.': 'Incorporated', 'Ltd.': 'Limited',
            'Corp.': 'Corporation',
        }
        
        self._audio_math_replacements = {
            r'\^2': ' squared',
            r'\^3': ' cubed',
            r'\^': ' to the power of ',
            r'√': 'square root of ',
            r'∑': 'sum of ',
            r'∫': 'integral of ',
            r'≈': 'approximately ',
            r'≤': 'less than or equal to ',
            r'≥': 'greater than or equal to ',
            r'≠': 'not equal to ',
            r'±': 'plus or minus ',
            r'×': ' times ',
            r'÷': ' divided by ',
            r'\+': ' plus ',
            r'=': ' equals ',
        }

    def _fix_encoding_issues(self, text: str, aggressive_ascii_filter: bool = False) -> str:
        """Fix common encoding issues and remove non-printables.

        Args:
            text: Input text
            aggressive_ascii_filter: If True, remove all non-ASCII characters (default: False)
                                    If False, only normalize common Unicode and remove control chars
        """
        # Apply direct replacements first (normalize common Unicode to ASCII equivalents)
        for old, new in self._encoding_map.items():
            text = text.replace(old, new)

        # Optional: Remove ALL non-ASCII characters (disabled by default to preserve international text)
        if aggressive_ascii_filter:
            text = re.sub(r'[^\x00-\x7F\n\t]', '', text)

        # Remove control characters (but keep printable Unicode if aggressive_ascii_filter=False)
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (tabs, newlines, spaces)."""
        text = text.replace('\t', ' ') # Replace tabs with spaces
        text = text.replace('\r\n', '\n').replace('\r', '\n') # Normalize line endings
        text = self._multi_newline_regex.sub('\n\n', text) # Reduce multiple blank lines
        text = self._multi_space_regex.sub(' ', text) # Reduce multiple spaces
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()

    def _fix_pdf_artifacts(self, text: str) -> str:
        """Fix common issues specific to PDF extraction (hyphens, spacing)."""
        text = self._pdf_hyphen_regex.sub(r'\1\2', text) # Fix hyphenation
        text = self._pdf_punct_space_regex.sub(r'\1', text) # Fix space before punctuation
        text = self._pdf_punct_word_regex.sub(r'\1 \2', text) # Fix space after punctuation
        text = self._pdf_page_num_regex.sub('\n', text) # Remove page numbers
        text = self._pdf_bullet_regex.sub('- ', text) # Normalize bullets
        return text

    def _segment_sentences(self, text: str) -> str:
        """
        Segments text into sentences using a rule-based approach.
        This is a basic implementation and might not cover all edge cases.
        """
        self.logger.debug("Segmenting sentences...")
        # Add a space after periods, question marks, and exclamation points if not already present
        text = re.sub(r'(?<=[.!?])(?=[^\s0-9])', r' ', text)
        # Split by sentence-ending punctuation followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty strings and re-join with a single space
        return ' '.join([s.strip() for s in sentences if s.strip()])

    def _remove_references_section(self, text: str) -> str:
        """
        Remove or move reference sections to the end of the text.

        Args:
            text: Input text

        Returns:
            Text with references section removed or relocated
        """
        # Try to find reference section using various patterns
        for pattern in self._ref_section_patterns:
            match = pattern.search(text)
            if match:
                # Found reference section - remove everything from this point onward
                # since references are typically at the end of research papers
                ref_start = match.start()
                main_content = text[:ref_start].strip()
                self.logger.debug(f"Removed references section starting at position {ref_start}")
                return main_content

        # No clear reference section found, return as-is
        return text

    def clean_for_audio(self, text: str, remove_references: bool = True) -> str:
        """
        Clean text specifically for audio/podcast generation.

        Args:
            text: Input text (likely already preprocessed for LLM).
            remove_references: Whether to remove/relocate reference sections (default: True).

        Returns:
            Cleaned text suitable for audio.
        """
        self.logger.debug("Applying audio-specific cleaning...")

        # Remove reference sections first (common in research papers)
        if remove_references:
            text = self._remove_references_section(text)

        # Convert URLs/Emails (might have been done in preprocess_for_llm, but run again)
        text = self._url_regex.sub('[web link]', text)
        text = self._email_regex.sub('[email address]', text)

        # Remove or simplify citations (more aggressive)
        text = self._citation_regex.sub('', text)  # Remove [1] style citations
        text = self._year_regex.sub('', text)      # Remove (2024) style years

        # Remove inline citations with author names (e.g., "Smith et al. (2020)")
        text = re.sub(r'\b\w+\s+et\s+al\.\s*\(\d{4}\)', '', text)

        # Remove standalone author-year citations (e.g., "(Smith, 2020)")
        text = re.sub(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*\d{4}\)', '', text)

        # Remove complex LaTeX
        text = self._latex_inline_regex.sub('[formula]', text)
        text = self._latex_block_regex.sub('[formula]', text)
        text = self._latex_cmd_regex.sub('[formula]', text)

        # Convert simple math symbols
        for pattern, replacement in self._audio_math_replacements.items():
            text = re.sub(pattern, replacement, text)

        # Remove tables (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        for line in lines:
            if '|' in line and line.count('|') > 2:
                if not in_table:
                    cleaned_lines.append('[table content omitted]')
                    in_table = True
            else:
                in_table = False
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)

        # Expand abbreviations
        # Use regex to ensure we match whole words (e.g., 'Mr.' not 'OMr.')
        for abbr, expansion in self._audio_abbreviations.items():
            # \b matches word boundary
            # re.escape handles the '.' in 'Mr.'
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text)

        # Remove special characters that don't belong in audio (but preserve punctuation)
        # This removes things like ©, ®, ™, §, ¶, †, ‡, etc.
        text = re.sub(r'[©®™§¶†‡°•◦▪▫■□●○▸▹►▻‣⁃←→↑↓]', '', text)

        # Clean up multiple consecutive punctuation marks
        text = re.sub(r'([.,!?])\1+', r'\1', text)  # e.g., "..." -> "."

        # Remove orphaned punctuation with excessive spacing
        text = re.sub(r'\s+([.,;!?])\s+', r'\1 ', text)

        # Final whitespace cleanup
        text = self._normalize_whitespace(text)

        return text

    def preprocess_for_llm(self,
                           text: str,
                           remove_urls: bool = True,
                           remove_emails: bool = True,
                           normalize_whitespace: bool = True,
                           fix_encoding: bool = True,
                           fix_pdf: bool = True,
                           segment_sentences: bool = False,
                           aggressive_ascii_filter: bool = False) -> str:
        """
        Preprocess text for LLM processing.

        Args:
            text: Input text.
            remove_urls: Whether to remove URLs.
            remove_emails: Whether to remove email addresses.
            normalize_whitespace: Whether to normalize whitespace.
            fix_encoding: Whether to fix encoding issues.
            fix_pdf: Whether to fix common PDF artifacts (hyphens, etc.).
            segment_sentences: Whether to segment text into sentences (disabled by default).
            aggressive_ascii_filter: Whether to remove all non-ASCII characters (disabled by default).

        Returns:
            Preprocessed text.
        """
        if fix_encoding:
            text = self._fix_encoding_issues(text, aggressive_ascii_filter=aggressive_ascii_filter)
            
        if remove_urls:
            text = self._url_regex.sub('[URL]', text)
            
        if remove_emails:
            text = self._email_regex.sub('[EMAIL REDACTED]', text)
        
        if fix_pdf:
            text = self._fix_pdf_artifacts(text)

        if segment_sentences: # New step
            text = self._segment_sentences(text)
            
        if normalize_whitespace:
            text = self._normalize_whitespace(text)

        return text


class SpeakerSegmentParser:
    """
    Parser for multi-speaker markdown segments.

    Extracts speaker-labeled text segments from markdown documents following
    the format: **[Speaker Name]:** text content

    This parser is designed to work with the output of the LLM-based markdown
    generation, which formats multi-speaker conversations with speaker labels.

    Example markdown format:
        **[Speaker Host]:** Welcome to the show!
        **[Speaker Guest]:** Thanks for have me.
        **[Speaker Host]:** Let's dive into the topic.

    Features:
    - Extracts speaker name and associated text
    - Handles multi-line speaker segments
    - Identifies unique speakers in document
    - Preserves original text content
    """

    # Regex pattern for speaker labels
    # Matches: **[Speaker Name]:** followed by text until next speaker or end
    SPEAKER_PATTERN = r'\*\*\[Speaker\s+([^\]]+)\]:\*\*\s*(.+?)(?=\n\*\*\[Speaker|$)'

    def __init__(self):
        """Initialize the speaker parser."""
        self.logger = get_logger_conf(f"{__name__}.SpeakerSegmentParser")

    def parse_speakers(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        Parse all speaker segments from markdown text.

        Args:
            markdown_text: Markdown document with speaker labels

        Returns:
            List of dicts with keys:
                - 'speaker': Speaker name (e.g., "Host", "Guest")
                - 'text': Text content for this segment
                - 'index': Sequential index in document

        Example:
            parser = SpeakerSegmentParser()
            segments = parser.parse_speakers(markdown_text)
            for seg in segments:
                print(f"{seg['speaker']}: {seg['text']}")
        """
        matches = re.findall(
            self.SPEAKER_PATTERN,
            markdown_text,
            re.DOTALL
        )

        segments = []
        for idx, (speaker, text) in enumerate(matches):
            segments.append({
                'speaker': speaker.strip(),
                'text': text.strip(),
                'index': idx
            })

        self.logger.debug(f"Parsed {len(segments)} speaker segments")
        return segments

    def get_unique_speakers(self, markdown_text: str) -> List[str]:
        """
        Get list of unique speakers in the document.

        Args:
            markdown_text: Markdown document with speaker labels

        Returns:
            List of unique speaker names in order of first appearance

        Example:
            speakers = parser.get_unique_speakers(markdown_text)
            # Returns: ["Host", "Guest", "Narrator"]
        """
        segments = self.parse_speakers(markdown_text)

        # Preserve order of first appearance
        seen = set()
        unique_speakers = []
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in seen:
                seen.add(speaker)
                unique_speakers.append(speaker)

        self.logger.info(f"Found {len(unique_speakers)} unique speakers: {unique_speakers}")
        return unique_speakers

    def count_speaker_segments(self, markdown_text: str) -> Dict[str, int]:
        """
        Count number of segments per speaker.

        Args:
            markdown_text: Markdown document with speaker labels

        Returns:
            Dict mapping speaker name to segment count

        Example:
            counts = parser.count_speaker_segments(markdown_text)
            # Returns: {"Host": 15, "Guest": 12, "Narrator": 3}
        """
        segments = self.parse_speakers(markdown_text)

        counts = {}
        for seg in segments:
            speaker = seg['speaker']
            counts[speaker] = counts.get(speaker, 0) + 1

        return counts

    def has_multiple_speakers(self, markdown_text: str) -> bool:
        """
        Check if document has multiple speakers.

        Args:
            markdown_text: Markdown document

        Returns:
            bool: True if 2+ unique speakers found
        """
        unique_speakers = self.get_unique_speakers(markdown_text)
        return len(unique_speakers) >= 2

    def get_speaker_turns(self, markdown_text: str) -> List[tuple[str, str]]:
        """
        Get sequential speaker turns as (speaker, text) tuples.

        Simpler interface than parse_speakers() for basic iteration.

        Args:
            markdown_text: Markdown document

        Returns:
            List of (speaker_name, text_content) tuples

        Example:
            for speaker, text in parser.get_speaker_turns(markdown_text):
                print(f"{speaker}: {text[:50]}...")
        """
        segments = self.parse_speakers(markdown_text)
        return [(seg['speaker'], seg['text']) for seg in segments]

    def replace_speaker_name(
        self,
        markdown_text: str,
        old_name: str,
        new_name: str
    ) -> str:
        """
        Replace all occurrences of a speaker name.

        Useful for renaming speakers in the document.

        Args:
            markdown_text: Original markdown
            old_name: Speaker name to replace
            new_name: New speaker name

        Returns:
            str: Modified markdown with renamed speaker

        Example:
            text = parser.replace_speaker_name(text, "Guest", "Expert")
        """
        pattern = rf'\*\*\[Speaker\s+{re.escape(old_name)}\]:\*\*'
        replacement = f'**[Speaker {new_name}]:**'
        return re.sub(pattern, replacement, markdown_text)

    def strip_speaker_labels(self, markdown_text: str) -> str:
        """
        Remove all speaker labels, leaving only text content.

        Args:
            markdown_text: Markdown with speaker labels

        Returns:
            str: Plain text with labels removed

        Example:
            plain_text = parser.strip_speaker_labels(markdown_text)
        """
        # Remove speaker labels but keep text
        pattern = r'\*\*\[Speaker\s+[^\]]+\]:\*\*\s*'
        return re.sub(pattern, '', markdown_text)