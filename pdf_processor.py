"""
PDF Processor Module
Handles PDF extraction, validation, and metadata processing
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import PyPDF2
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for better extraction
import re

from logging_config import get_logger, log_execution_time, LoggingProgress
from config import MAX_PDF_SIZE_MB, MAX_CHARS_PER_FILE, SUPPORTED_FORMATS

logger = get_logger(__name__)


@dataclass
class PDFMetadata:
    """PDF metadata container"""
    file_path: Path
    num_pages: int
    title: Optional[str]
    author: Optional[str]
    subject: Optional[str]
    creation_date: Optional[str]
    modification_date: Optional[str]
    file_size_mb: float
    raw_metadata: Dict[str, Any]


@dataclass
class ExtractionResult:
    """Result of PDF text extraction"""
    text: str
    metadata: PDFMetadata
    page_texts: List[str]
    extraction_method: str
    warnings: List[str]
    char_count: int
    word_count: int


class PDFProcessor:
    """Advanced PDF processing with multiple extraction methods"""
    
    def __init__(self, 
                 max_chars: int = MAX_CHARS_PER_FILE,
                 max_size_mb: float = MAX_PDF_SIZE_MB,
                 preserve_layout: bool = False):
        """
        Initialize PDF processor
        
        Args:
            max_chars: Maximum characters to extract
            max_size_mb: Maximum PDF file size in MB
            preserve_layout: Whether to preserve layout during extraction
        """
        self.max_chars = max_chars
        self.max_size_mb = max_size_mb
        self.preserve_layout = preserve_layout
        logger.info(f"Initialized PDFProcessor (max_chars={max_chars}, max_size_mb={max_size_mb})")
        
    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        # Check existence
        if not file_path.exists():
            return False, f"File not found: {file_path}"
            
        # Check extension
        if file_path.suffix.lower() != '.pdf':
            return False, f"Not a PDF file: {file_path.suffix}"
            
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_size_mb:
            return False, f"File too large: {file_size_mb:.1f}MB > {self.max_size_mb}MB"
            
        # Try to open as PDF
        try:
            with open(file_path, 'rb') as f:
                PdfReader(f)
        except Exception as e:
            return False, f"Invalid PDF: {str(e)}"
            
        return True, ""
        
    @log_execution_time()
    def extract_text(self, file_path: Path) -> Optional[ExtractionResult]:
        """
        Extract text from PDF using multiple methods
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ExtractionResult or None if extraction fails
        """
        file_path = Path(file_path)
        
        # Validate file
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            logger.error(f"Validation failed: {error_msg}")
            return None
            
        logger.info(f"Extracting text from: {file_path.name}")
        
        # Try PyMuPDF first (usually better quality)
        result = self._extract_with_pymupdf(file_path)
        
        # Fallback to PyPDF2 if needed
        if result is None or len(result.text.strip()) < 100:
            logger.info("PyMuPDF extraction insufficient, trying PyPDF2")
            result = self._extract_with_pypdf2(file_path)
            
        if result:
            # Post-process the extracted text
            result = self._post_process_extraction(result)
            logger.info(f"Extraction complete: {result.char_count:,} chars, {result.word_count:,} words")
            
        return result
        
    def _extract_with_pymupdf(self, file_path: Path) -> Optional[ExtractionResult]:
        """Extract text using PyMuPDF (fitz)"""
        try:
            import fitz
            
            # Get metadata first
            metadata = self._extract_metadata(file_path)
            if not metadata:
                return None
                
            doc = fitz.open(file_path)
            page_texts = []
            total_text = []
            warnings = []
            total_chars = 0
            
            with LoggingProgress(logger, f"Extracting {doc.page_count} pages", doc.page_count) as progress:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    # Extract text with layout preservation if requested
                    if self.preserve_layout:
                        text = page.get_text("text", sort=True)
                    else:
                        text = page.get_text()
                        
                    # Check character limit
                    if total_chars + len(text) > self.max_chars:
                        remaining = self.max_chars - total_chars
                        text = text[:remaining]
                        warnings.append(f"Reached character limit at page {page_num + 1}")
                        page_texts.append(text)
                        total_text.append(text)
                        break
                        
                    page_texts.append(text)
                    total_text.append(text)
                    total_chars += len(text)
                    
                    progress.update(1, f"Page {page_num + 1}")
                    
            doc.close()
            
            full_text = '\n'.join(total_text)
            
            return ExtractionResult(
                text=full_text,
                metadata=metadata,
                page_texts=page_texts,
                extraction_method="PyMuPDF",
                warnings=warnings,
                char_count=len(full_text),
                word_count=len(full_text.split())
            )
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return None
            
    def _extract_with_pypdf2(self, file_path: Path) -> Optional[ExtractionResult]:
        """Extract text using PyPDF2"""
        try:
            # Get metadata first
            metadata = self._extract_metadata(file_path)
            if not metadata:
                return None
                
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                page_texts = []
                total_text = []
                warnings = []
                total_chars = 0
                
                num_pages = len(pdf_reader.pages)
                
                with LoggingProgress(logger, f"Extracting {num_pages} pages", num_pages) as progress:
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        # Check character limit
                        if total_chars + len(text) > self.max_chars:
                            remaining = self.max_chars - total_chars
                            text = text[:remaining]
                            warnings.append(f"Reached character limit at page {page_num + 1}")
                            page_texts.append(text)
                            total_text.append(text)
                            break
                            
                        page_texts.append(text)
                        total_text.append(text)
                        total_chars += len(text)
                        
                        progress.update(1, f"Page {page_num + 1}")
                        
            full_text = '\n'.join(total_text)
            
            return ExtractionResult(
                text=full_text,
                metadata=metadata,
                page_texts=page_texts,
                extraction_method="PyPDF2",
                warnings=warnings,
                char_count=len(full_text),
                word_count=len(full_text.split())
            )
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return None
            
    def _extract_metadata(self, file_path: Path) -> Optional[PDFMetadata]:
        """Extract PDF metadata"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                raw_metadata = pdf_reader.metadata or {}
                
                # Extract specific fields
                metadata = PDFMetadata(
                    file_path=file_path,
                    num_pages=len(pdf_reader.pages),
                    title=raw_metadata.get('/Title', None),
                    author=raw_metadata.get('/Author', None),
                    subject=raw_metadata.get('/Subject', None),
                    creation_date=str(raw_metadata.get('/CreationDate', '')),
                    modification_date=str(raw_metadata.get('/ModDate', '')),
                    file_size_mb=file_path.stat().st_size / (1024 * 1024),
                    raw_metadata=dict(raw_metadata)
                )
                
                return metadata
                
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return None
            
    def _post_process_extraction(self, result: ExtractionResult) -> ExtractionResult:
        """Post-process extracted text"""
        text = result.text
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common extraction issues
        text = self._fix_common_issues(text)
        
        # Update result
        result.text = text
        result.char_count = len(text)
        result.word_count = len(text.split())
        
        return result
        
    def _fix_common_issues(self, text: str) -> str:
        """Fix common PDF extraction issues"""
        # Fix hyphenation at line ends
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Fix spacing issues around punctuation
        text = re.sub(r'\s+([.,;!?])', r'\1', text)
        text = re.sub(r'([.,;!?])(\w)', r'\1 \2', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\nPage \d+ of \d+\n', '\n', text)
        
        # Fix bullet points
        text = re.sub(r'^[•·■□▪▫◦‣⁃]', '-', text, flags=re.MULTILINE)
        
        return text


class BatchPDFProcessor:
    """Process multiple PDF files"""
    
    def __init__(self, processor: Optional[PDFProcessor] = None):
        """
        Initialize batch processor
        
        Args:
            processor: PDFProcessor instance to use
        """
        self.processor = processor or PDFProcessor()
        self.logger = get_logger(f"{__name__}.BatchProcessor")
        
    def process_directory(self, 
                         directory: Path,
                         pattern: str = "*.pdf",
                         recursive: bool = False) -> List[ExtractionResult]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Directory to process
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of extraction results
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"Directory not found: {directory}")
            return []
            
        # Find PDF files
        if recursive:
            pdf_files = list(directory.rglob(pattern))
        else:
            pdf_files = list(directory.glob(pattern))
            
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_file in pdf_files:
            self.logger.info(f"Processing: {pdf_file.name}")
            result = self.processor.extract_text(pdf_file)
            
            if result:
                results.append(result)
            else:
                self.logger.warning(f"Failed to process: {pdf_file.name}")
                
        self.logger.info(f"Processed {len(results)}/{len(pdf_files)} files successfully")
        return results
        
    def process_files(self, file_paths: List[Path]) -> List[ExtractionResult]:
        """
        Process a list of PDF files
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of extraction results
        """
        results = []
        
        with LoggingProgress(self.logger, "Processing PDFs", len(file_paths)) as progress:
            for file_path in file_paths:
                file_path = Path(file_path)
                
                result = self.processor.extract_text(file_path)
                if result:
                    results.append(result)
                    progress.update(1, f"Processed: {file_path.name}")
                else:
                    self.logger.warning(f"Failed to process: {file_path.name}")
                    progress.update(1, f"Failed: {file_path.name}")
                    
        return results


class PDFTextCleaner:
    """Clean and preprocess PDF text for specific use cases"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.TextCleaner")
        
    def clean_for_audio(self, text: str) -> str:
        """
        Clean text specifically for audio/podcast generation
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text suitable for audio
        """
        # Remove or convert elements that don't work in audio
        
        # Convert URLs to spoken form
        text = re.sub(r'https?://[\S]+', '[web link]', text)
        
        # Convert email addresses
        text = re.sub(r'\S+@\S+\.\S+', '[email address]', text)
        
        # Remove or simplify citations
        text = re.sub(r'\[\d+\]', '', text)  # Remove [1] style citations
        text = re.sub(r'\(\d{4}\)', '', text)  # Remove (2024) style years
        
        # Convert mathematical expressions
        text = self._convert_math_for_speech(text)
        
        # Remove tables (they don't work well in audio)
        text = self._remove_tables(text)
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        return text
        
    def _convert_math_for_speech(self, text: str) -> str:
        """Convert mathematical expressions for speech"""
        # Simple conversions
        replacements = {
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
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
            
        # Remove complex LaTeX that can't be spoken
        text = re.sub(r'\$.*?\$', '[mathematical expression]', text)
        text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '[formula]', text)
        
        return text
        
    def _remove_tables(self, text: str) -> str:
        """Remove or summarize tables"""
        # Simple heuristic: lines with multiple | or tabs are likely tables
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line and line.count('|') > 2:
                if not in_table:
                    cleaned_lines.append('[Table content omitted]')
                    in_table = True
            else:
                in_table = False
                cleaned_lines.append(line)
                
        return '\n'.join(cleaned_lines)
        
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations for speech"""
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'Sr.': 'Senior',
            'Jr.': 'Junior',
            'Ph.D.': 'PhD',
            'M.D.': 'MD',
            'B.S.': 'Bachelor of Science',
            'M.S.': 'Master of Science',
            'i.e.': 'that is',
            'e.g.': 'for example',
            'etc.': 'et cetera',
            'vs.': 'versus',
            'Inc.': 'Incorporated',
            'Ltd.': 'Limited',
            'Corp.': 'Corporation',
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
            
        return text
