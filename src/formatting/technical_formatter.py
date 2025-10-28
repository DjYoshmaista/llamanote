# llamanote/formatting/technical_formatter.py
"""
Technical Formatter Module
Specialized formatter for technical documents, focusing on structure, code, and lists.
"""

import re
from typing import List, Tuple

from .base_formatter import BaseFormatter, PatternReplacer
from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

class TechnicalFormatter(BaseFormatter):
    """
    Formats text into a structured technical document, identifying headers,
    code blocks, lists, and important callouts.
    """
    
    def __init__(self):
        super().__init__(style="technical")
        
        # Patterns for identifying inline code
        self.inline_code_patterns = [
            # Words with underscores, common in programming (e.g., var_name)
            (r'\b([a-z0-9_]+_[a-z0-9_]+)\b', r'`\1`'),
            # Function calls (e.g., function_name())
            (r'\b([a-zA-Z0-9_]+\s*\(\))\b', r'`\1`'),
            # File extensions (e.g., .json, .py)
            (r'(\.[a-zA-Z0-9]{2,5})\b', r'`\1`'),
            # Simple variable names in camelCase or PascalCase (heuristic)
            (r'\b([a-z]+[A-Z][a-zA-Z0-9]+)\b', r'`\1`'),
            (r'\b([A-Z][a-z]+[A-Z][a-zA-Z0-9]+)\b', r'`\1`'),
        ]
        self.inline_code_replacer = PatternReplacer(self.inline_code_patterns)
        
        # Patterns for callouts (Note, Warning, etc.)
        self.note_patterns = [
            (r'^(Note|NOTE):\s*(.+)', '📝 **Note:** \\2'),
            (r'^(Warning|WARNING):\s*(.+)', '⚠️ **Warning:** \\2'),
            (r'^(Tip|TIP):\s*(.+)', '💡 **Tip:** \\2'),
            (r'^(Important|IMPORTANT):\s*(.+)', '❗ **Important:** \\2'),
        ]
        self.note_replacer = PatternReplacer(self.note_patterns)
        
    def format(self, text: str, **kwargs) -> str:
        """
        Overrides base format method to apply technical formatting.

        Args:
            text: The text to format.
            **kwargs:
                auto_generate_toc (bool): Whether to generate a Table of Contents.
                add_structure (bool): Whether to add headers/titles.

        Returns:
            Formatted technical markdown string.
        """
        # Preserve code blocks first (handled by base class)
        text, code_blocks = self._extract_code_blocks(text)
        
        # Identify and format sections
        sections, non_section_text = self._identify_sections(text)
        
        formatted_sections = []
        toc_entries = []
        
        # Add any text before the first section
        if non_section_text:
             formatted_sections.append(self._format_technical_content(non_section_text))

        for i, (title, content) in enumerate(sections):
            # Format section title
            formatted_title = self._format_header(title)
            formatted_sections.append(formatted_title)
            
            # Create ToC entry (simple version)
            slug = re.sub(r'[^a-z0-9\s-]', '', title.lower()).strip().replace(' ', '-')
            toc_entries.append(f"{'  '*(len(formatted_title.split(' ')[0])-2)}- [{title}](#{slug})")
                
            # Format section content
            formatted_content = self._format_technical_content(content)
            formatted_sections.append(formatted_content)
            
        # Combine sections
        formatted_text = '\n\n'.join(formatted_sections)
        
        # Add TOC if requested
        if kwargs.get('auto_generate_toc', True) and toc_entries:
            toc = "## Table of Contents\n\n" + '\n'.join(toc_entries)
            formatted_text = toc + '\n\n---\n\n' + formatted_text
            
        # Restore code blocks
        formatted_text = self._restore_code_blocks(formatted_text, code_blocks)
        
        # Add main document title if it doesn't have one
        if kwargs.get('add_structure', True) and not formatted_text.lstrip().startswith('# '):
             formatted_text = f"# Processed Document\n\n{formatted_text}"
            
        return formatted_text.strip()

    def _format_header(self, title: str) -> str:
        """Determines header level based on title conventions."""
        # Simple heuristic: "Chapter X" or "Section X" is H2, others H3
        if re.match(r'^(chapter|section)\s+\d+', title.lower()):
            return f"## {title}"
        # Titles in ALL CAPS or Title Case are likely major sections
        if title.isupper() or title.istitle():
            return f"## {title}"
        # Default to H3 for sub-sections
        return f"### {title}"

    def _identify_sections(self, text: str) -> Tuple[List[Tuple[str, str]], str]:
        """Identify logical sections based on common header-like lines."""
        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sections = []
        current_content = []
        current_title = ""
        pre_content = [] # Content before the first header

        # Regex to identify potential headers (short, no punctuation at end, often title-cased)
        # This is a heuristic and may misclassify.
        header_regex = re.compile(r'^[A-Za-z0-9\s_\-:]+$')

        for para in paragraphs:
            is_header = False
            # Check if paragraph looks like a header
            if (len(para) < 100 and 
                para.strip() and 
                not para.strip().endswith(('.', '!', '?', ':', ',')) and 
                header_regex.match(para)):
                
                # Check if it has a mix of case (not just "THIS IS A SENTENCE")
                if not (para.isupper() or para.islower()):
                    is_header = True
                # Allow all-caps if short
                elif para.isupper() and len(para.split()) < 7:
                    is_header = True

            if is_header:
                # If we find a header, save the previous section
                if current_title or current_content:
                    sections.append((current_title, '\n\n'.join(current_content)))
                elif not sections: # This is the first header, save pre-content
                     pre_content = current_content
                
                current_title = para
                current_content = []
            else:
                # This is content
                current_content.append(para)
                
        # Add the final section
        if current_title or current_content:
            if not sections and not pre_content: # No headers found at all
                 pre_content = current_content
            else:
                 sections.append((current_title, '\n\n'.join(current_content)))

        return sections, '\n\n'.join(pre_content)

    def _format_technical_content(self, content: str) -> str:
        """Apply formatting to the body of a technical section."""
        
        # Format lists (simple regex-based)
        content = self._format_lists(content)
        
        # Format callouts (Notes, Warnings)
        content = self.note_replacer.replace(content)
        
        # Format inline code
        # Be careful not to apply inside existing code blocks (which should be extracted)
        content = self.inline_code_replacer.replace(content)
        
        return content
        
    def _format_lists(self, text: str) -> str:
        """Format bullet points and numbered lists."""
        lines = text.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            stripped_line = line.strip()
            # Match numbered lists (1., 1), a., a))
            num_match = re.match(r'^(\d+[\.\)]|[a-zA-Z][\.\)])\s+', stripped_line)
            # Match bullet lists (*, -, +)
            bullet_match = re.match(r'^[\*\-+]+\s+', stripped_line)

            if num_match:
                # Standardize to "1. "
                item_marker = num_match.group(1)
                item_content = stripped_line[len(item_marker):].strip()
                # Simple re-numbering isn't safe, just ensure formatting
                # For simplicity, we'll just standardize the marker
                marker = f"1. " # This is naive, but better than nothing
                if item_marker.isalpha(): marker = "a. "
                
                formatted_lines.append(f"{' '*4}{marker}{item_content}") # Indent for clarity
                in_list = True
            elif bullet_match:
                # Standardize to "- "
                item_marker = bullet_match.group(0)
                item_content = stripped_line[len(item_marker):].strip()
                
                formatted_lines.append(f"{' '*4}{'- '}{item_content}") # Indent
                in_list = True
            else:
                # Not a list item
                if in_list: # Add extra newline after list
                     formatted_lines.append("")
                formatted_lines.append(line) # Keep original line (might be blank)
                in_list = False
                
        return '\n'.join(formatted_lines)
