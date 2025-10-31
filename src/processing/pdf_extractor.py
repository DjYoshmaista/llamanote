# llamanote/processing/pdf_extractor.py
"""
PDF Processor Module
Handles PDF validation, metadata extraction, and text extraction using strategies.
"""

import abc
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import fitz  # PyMuPDF
from dotenv import load_dotenv

from ..utils.logger import get_logger_conf, LoggingProgress
from ..utils.decorators import log_execution_time
from ..utils.validators import validate_file_path
from ..config.settings import MAX_PDF_SIZE_MB, MAX_CHARS_PER_FILE
from ..core.types import PDFMetadata, ExtractionResult

dotenv_path = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=dotenv_path)
logger = get_logger_conf(__name__)

SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS")
if SUPPORTED_FORMATS:
    ext_list = SUPPORTED_FORMATS.split(',')
    extensions = tuple(ext_list)
    print(f"Original string: {SUPPORTED_FORMATS}")
    print(f"Resulting tuple: {extensions}")
    print(f"Type of 'extensions': {type(extensions)}")
else:
    print("SUPPORTED_FORMATS environment variable not found.")

# --- Extraction Strategy Interface ---
class BaseExtractionStrategy(abc.ABC):
    """Abstract base class for a PDF text extraction strategy."""
    
    def __init__(self, preserve_layout: bool = False, max_chars: int = MAX_CHARS_PER_FILE):
        self.preserve_layout = preserve_layout
        self.max_chars = max_chars
        self.logger = get_logger_conf(f"{__name__}.{self.__class__.__name__}")

    @abc.abstractmethod
    def extract(self, file_path: Path, num_pages: int) -> Tuple[str, List[str], List[str]]:
        """
        Extracts text from the PDF.
        Returns: (full_text, list_of_page_texts, list_of_warnings)
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the extraction method."""
        pass

# --- Concrete Strategies ---

class PyMuPDFStrategy(BaseExtractionStrategy):
    """Extracts text using PyMuPDF (fitz), which is generally more accurate."""

    @property
    def name(self) -> str:
        return "PyMuPDF"

    def extract(self, file_path: Path, num_pages: int) -> Tuple[str, List[str], List[str]]:
        doc: Optional[fitz.Document] = None
        page_texts: List[str] = []
        total_text_builder: List[str] = []
        warnings: List[str] = []
        total_chars = 0
        
        try:
            doc = fitz.open(file_path)
            # num_pages = doc.page_count # Already provided, but can double-check
            
            with LoggingProgress(self.logger, f"Extracting {num_pages} pages (PyMuPDF)", num_pages) as progress:
                for page_num in range(num_pages):
                    page = doc[page_num]
                    
                    # 'text' preserves basic layout, 'blocks' gives more structure
                    text_mode = "text" if self.preserve_layout else "simple"
                    text = page.get_text(text_mode, sort=self.preserve_layout)
                        
                    if total_chars + len(text) > self.max_chars:
                        remaining = self.max_chars - total_chars
                        text = text[:remaining]
                        warnings.append(f"Reached character limit ({self.max_chars}) at page {page_num + 1}")
                        page_texts.append(text)
                        total_text_builder.append(text)
                        total_chars += len(text)
                        break # Stop processing
                        
                    page_texts.append(text)
                    total_text_builder.append(text)
                    total_chars += len(text)
                    
                    progress.update(1, f"Page {page_num + 1}")
            
            full_text = '\n\n'.join(total_text_builder) # Join pages with double newline
            return full_text, page_texts, warnings

        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {e}", exc_info=True)
            warnings.append(f"PyMuPDF failed: {e}")
            # Return any text extracted so far
            full_text = '\n\n'.join(total_text_builder)
            return full_text, page_texts, warnings
        finally:
            if doc:
                doc.close()

class PyPDF2Strategy(BaseExtractionStrategy):
    """Extracts text using PyPDF2, as a fallback."""

    @property
    def name(self) -> str:
        return "PyPDF2"

    def extract(self, file_path: Path, num_pages: int) -> Tuple[str, List[str], List[str]]:
        page_texts: List[str] = []
        total_text_builder: List[str] = []
        warnings: List[str] = []
        total_chars = 0
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                # num_pages = len(pdf_reader.pages) # Already provided
                
                with LoggingProgress(self.logger, f"Extracting {num_pages} pages (PyPDF2)", num_pages) as progress:
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text is None: text = "" # Handle empty pages
                        
                        if total_chars + len(text) > self.max_chars:
                            remaining = self.max_chars - total_chars
                            text = text[:remaining]
                            warnings.append(f"Reached character limit ({self.max_chars}) at page {page_num + 1}")
                            page_texts.append(text)
                            total_text_builder.append(text)
                            total_chars += len(text)
                            break
                            
                        page_texts.append(text)
                        total_text_builder.append(text)
                        total_chars += len(text)
                        progress.update(1, f"Page {page_num + 1}")
                        
            full_text = '\n'.join(total_text_builder) # Join pages with single newline
            return full_text, page_texts, warnings
            
        except Exception as e:
            self.logger.error(f"PyPDF2 extraction failed: {e}", exc_info=True)
            warnings.append(f"PyPDF2 failed: {e}")
            full_text = '\n'.join(total_text_builder)
            return full_text, page_texts, warnings

# --- Metadata Extractor (Refactoring Item 9) ---

class MetadataExtractor:
    """Extracts PDF metadata using PyPDF2."""
    
    @staticmethod
    def extract_metadata(file_path: Path) -> Optional[PDFMetadata]:
        """Extracts PDF metadata."""
        logger.debug(f"Extracting metadata from: {file_path.name}")
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file, strict=False) # Be lenient
                
                raw_metadata = pdf_reader.metadata or {}
                
                # Helper to safely get metadata values
                def get_meta_value(key: str) -> Optional[str]:
                    val = raw_metadata.get(key)
                    if val is None: return None
                    return str(val).strip()

                metadata = PDFMetadata(
                    file_path=file_path,
                    num_pages=len(pdf_reader.pages),
                    title=get_meta_value('/Title'),
                    author=get_meta_value('/Author'),
                    subject=get_meta_value('/Subject'),
                    creation_date=get_meta_value('/CreationDate'),
                    modification_date=get_meta_value('/ModDate'),
                    file_size_mb=file_path.stat().st_size / (1024 * 1024),
                    raw_metadata=dict(raw_metadata) # Store raw dict
                )
                
                return metadata
                
        except PdfReadError as e:
            logger.warning(f"Could not read PDF metadata (file may be corrupt): {file_path.name}. Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Metadata extraction failed for {file_path.name}: {e}", exc_info=True)
            return None


# --- Main PDF Processor Class (Refactored) ---

class PDFProcessor:
    """
    Validates PDF, extracts metadata, and extracts text using a
    primary strategy (PyMuPDF) with a fallback (PyPDF2).
    """
    
    def __init__(self, 
                 max_chars: int = MAX_CHARS_PER_FILE,
                 max_size_mb: float = MAX_PDF_SIZE_MB,
                 preserve_layout: bool = False):
        """
        Initialize PDF processor.
        
        Args:
            max_chars: Maximum characters to extract.
            max_size_mb: Maximum PDF file size in MB.
            preserve_layout: Whether to preserve layout during extraction (passed to strategy).
        """
        self.max_chars = max_chars
        self.max_size_mb = max_size_mb
        self.preserve_layout = preserve_layout
        
        # Initialize strategies
        self.primary_strategy = PyMuPDFStrategy(preserve_layout, max_chars)
        self.fallback_strategy = PyPDF2Strategy(preserve_layout, max_chars)
        
        logger.info(f"Initialized PDFProcessor (Max Chars={max_chars}, Max Size={max_size_mb}MB)")
        
    @log_execution_time(logger_name="llamanote.processing.pdf_extractor")
    def extract_text(self, file_path: Path) -> Optional[ExtractionResult]:
        """
        Extract text from PDF using the best available method.
        
        Args:
            file_path: Path to PDF file.
            
        Returns:
            ExtractionResult or None if extraction fails.
        """
        file_path = Path(file_path)
        
        # 1. Validate file
        is_valid, error_msg = validate_file_path(
            file_path,
            check_existence=True,
            allowed_extensions=['.pdf'],
            max_size_mb=self.max_size_mb
        )
        if not is_valid:
            logger.error(f"PDF validation failed for {file_path.name}: {error_msg}")
            return None
            
        logger.info(f"Extracting text from: {file_path.name}")
        
        # 2. Extract Metadata
        metadata = MetadataExtractor.extract_metadata(file_path)
        if not metadata:
            logger.warning(f"Could not extract metadata from {file_path.name}. Proceeding with extraction.")
            # Create a placeholder metadata object if extraction failed
            metadata = PDFMetadata(
                 file_path=file_path,
                 num_pages=0, # Unknown
                 file_size_mb=file_path.stat().st_size / (1024 * 1024)
            )
            # Try to get page count from PyMuPDF even if metadata fails
            try:
                 with fitz.open(file_path) as doc:
                      metadata.num_pages = doc.page_count
            except Exception:
                 logger.warning(f"Could not determine page count for {file_path.name}.")
                 return None # Fail fast if we can't even open it

        if metadata.num_pages == 0:
             logger.warning(f"PDF {file_path.name} has 0 pages. No text extracted.")
             return ExtractionResult(
                 text="", metadata=metadata, page_texts=[], extraction_method="none",
                 warnings=["File contains 0 pages."], char_count=0, word_count=0
             )

        # 3. Try Primary Strategy (PyMuPDF)
        full_text, page_texts, warnings = self.primary_strategy.extract(file_path, metadata.num_pages)
        method_used = self.primary_strategy.name
        
        # 4. Try Fallback Strategy (PyPDF2) if primary failed or got little text
        min_text_threshold = 100 # Arbitrary threshold for "sufficient" text
        if not full_text or len(full_text.strip()) < min_text_threshold:
            logger.info(f"Primary strategy (PyMuPDF) extracted minimal text ({len(full_text)} chars). "
                        f"Attempting fallback (PyPDF2)...")
            
            fallback_text, fb_page_texts, fb_warnings = self.fallback_strategy.extract(file_path, metadata.num_pages)
            warnings.extend(fb_warnings)
            
            if len(fallback_text.strip()) > len(full_text.strip()):
                logger.info(f"Fallback strategy (PyPDF2) yielded more text ({len(fallback_text)} chars). Using fallback.")
                full_text = fallback_text
                page_texts = fb_page_texts
                method_used = f"{self.primary_strategy.name} (failed) -> {self.fallback_strategy.name}"
            else:
                 logger.info("Primary strategy (PyMuPDF) result was better or equal. Keeping primary result.")
                 method_used = self.primary_strategy.name # Stick with primary
        
        if not full_text:
            logger.error(f"All extraction methods failed for {file_path.name}.")
            return None
            
        # 5. Create Result
        char_count = len(full_text)
        word_count = len(full_text.split())

        logger.info(f"Extraction complete using {method_used}: {char_count:,} chars, {word_count:,} words")

        return ExtractionResult(
            text=full_text,
            metadata=metadata,
            page_texts=page_texts,
            extraction_method=method_used,
            warnings=warnings,
            char_count=char_count,
            word_count=word_count
        )
