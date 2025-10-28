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

    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues and remove non-printables."""
        for old, new in self._encoding_map.items():
            text = text.replace(old, new)
        
        # Remove other non-printable characters except for newline/tab
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

    def clean_for_audio(self, text: str) -> str:
        """
        Clean text specifically for audio/podcast generation.
        
        Args:
            text: Input text (likely already preprocessed for LLM).
            
        Returns:
            Cleaned text suitable for audio.
        """
        self.logger.debug("Applying audio-specific cleaning...")
        
        # Convert URLs/Emails (might have been done in preprocess_for_llm, but run again)
        text = self._url_regex.sub('[web link]', text)
        text = self._email_regex.sub('[email address]', text)
        
        # Remove or simplify citations
        text = self._citation_regex.sub('', text) # Remove [1] style citations
        text = self._year_regex.sub('', text)     # Remove (2024) style years

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

        # Final whitespace cleanup
        text = self._normalize_whitespace(text)
        
        return text

    def preprocess_for_llm(self, 
                           text: str,
                           remove_urls: bool = True,
                           remove_emails: bool = True,
                           normalize_whitespace: bool = True,
                           fix_encoding: bool = True,
                           fix_pdf: bool = True) -> str:
        """
        Preprocess text for LLM processing.

        Args:
            text: Input text.
            remove_urls: Whether to remove URLs.
            remove_emails: Whether to remove email addresses.
            normalize_whitespace: Whether to normalize whitespace.
            fix_encoding: Whether to fix encoding issues.
            fix_pdf: Whether to fix common PDF artifacts (hyphens, etc.).
            
        Returns:
            Preprocessed text.
        """
        if fix_encoding:
            text = self._fix_encoding_issues(text)
            
        if remove_urls:
            text = self._url_regex.sub('[https://www.youtube.com/@RedactedNews](https://www.youtube.com/@RedactedNews)', text)
            
        if remove_emails:
            text = self._email_regex.sub('[EMAIL REDACTED]', text)
        
        if fix_pdf:
            text = self._fix_pdf_artifacts(text)
            
        if normalize_whitespace:
            text = self._normalize_whitespace(text)
            
        return text
