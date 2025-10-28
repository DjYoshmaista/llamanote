#
# LlamaNote Enhanced
#
# A modular framework for processing documents (like PDFs)
# into structured, conversational, or narrative formats using LLMs.
#
# (This file makes 'llamanote' a Python package)
#

__version__ = "0.0.2-simplified"
__author__ = "LlamaNote Team (based on original concept)"

# Expose key components for easier import if needed
try:
    from .core.pipeline import ProcessingPipeline
    from .core.types import ProcessingMode, PipelineConfig, AudioConfig
    from .config.manager import ConfigManager
    from .models.registry import get_registry, get_model_entry
    from .models.backends.base import LLMBackend, AudioBackend
except ImportError as e:
    # This can happen during initial setup or if dependencies are missing
    import warnings
    warnings.warn(f"LlamaNote components could not be imported: {e}", ImportWarning)

# Initialize the logger for the package
from .utils.logger import setup_logging
setup_logging()
