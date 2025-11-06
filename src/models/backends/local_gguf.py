# llamanote/models/backends/local_gguf.py
"""
Llama.cpp Backend Integration
Provides support for GGUF quantized models via llama-cpp-python
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import re
import requests
from tqdm import tqdm
import sys

# --- LlamaNote Modules ---
from ...utils.logger import get_logger_conf, ConsoleOutput
from ...utils.decorators import log_execution_time
from ...utils.helpers import cleanup_resources, is_llama_cpp_installed
from ..hyperparameters import HyperparameterConfig
from ...config.manager import ConfigManager # Import ConfigManager
from ...config.settings import DEFAULT_CACHE_DIR, DEFAULT_GPU_LAYERS
from ...core.types import GenerationResult
from ...core.errors import ModelLoadError, GenerationError, ConfigurationError
from .base import LLMBackend # Import the abstract base class
from .mapper import HyperparamMapper # Import the mapper

# Check if llama-cpp-python is installed
LLAMACPP_AVAILABLE = is_llama_cpp_installed()

if LLAMACPP_AVAILABLE:
    from llama_cpp import Llama, LlamaGrammar
else:
    # Define dummy types for type hinting if import fails
    Llama = type('Llama', (object,), {})
    LlamaGrammar = type('LlamaGrammar', (object,), {})

logger = get_logger_conf(__name__)

# --- GGUF Model Manager (Helper Class) ---

class GGUFModelManager:
    """Manager for finding and inspecting GGUF models."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.cache_dir = self.config_manager.get_dir("model_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger_conf(f"{__name__}.GGUFManager")

    def list_available_models(self) -> List[Path]:
        """List available GGUF models in the cache directory."""
        patterns = ["*.gguf"] # Only support GGUF
        models = []
        for pattern in patterns:
            models.extend(list(self.cache_dir.glob(pattern)))
        models = [p for p in models if p.is_file()]
        return sorted(models)

    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get basic information about a GGUF model file."""
        if not model_path.exists() or not model_path.is_file():
            return {"error": "File not found"}

        info = {
            "path": str(model_path.resolve()),
            "name": model_path.name,
            "size_mb": model_path.stat().st_size / (1024 * 1024),
            "format": "gguf",
        }
        # Try to infer quantization from filename (common patterns)
        name_lower = model_path.name.lower()
        quant_match = re.search(r"[._-](q\d_[0-9kmls]|f(?:16|32))[._-]", name_lower, re.IGNORECASE)
        if quant_match:
            info["quantization_inferred"] = quant_match.group(1).upper()

        return info

    @log_execution_time(logger_name=__name__)
    def download_model_from_hf(
        self,
        repo_id: str,
        filename: str,
        revision: Optional[str] = "main"
    ) -> Optional[Path]:
        """Downloads a specific GGUF file from a Hugging Face repo."""
        if not LLAMACPP_AVAILABLE:
            self.logger.error("Cannot download GGUF from HF: llama-cpp-python is required for hf_hub_download.")
            ConsoleOutput.error("llama-cpp-python is required to download GGUF models.")
            return None
            
        from llama_cpp.llama_utils import hf_hub_download as llama_hf_download # Use llama-cpp's wrapper

        self.logger.info(f"Downloading GGUF: {repo_id} (file: {filename})")
        ConsoleOutput.info(f"Downloading {filename}...")
        try:
            downloaded_path_str = llama_hf_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
                resume_download=True,
                # Note: llama_cpp's hf_hub_download doesn't have a built-in progress bar
                # that we can easily hook into tqdm for the console.
            )
            downloaded_path = Path(downloaded_path_str)
            
            self.logger.info(f"Model downloaded to: {downloaded_path}")
            ConsoleOutput.success(f"Downloaded: {downloaded_path.name}")
            return downloaded_path
            
        except Exception as e:
            self.logger.error(f"Failed to download GGUF model {filename} from {repo_id}: {e}", exc_info=True)
            ConsoleOutput.error(f"Download failed: {e}")
            return None


# --- GGUF Backend Configuration ---

@dataclass
class GGUFConfig:
    """Configuration for llama.cpp backend"""
    # Model loading
    model_path: Path
    n_ctx: int = 4096  # Context size (default increased)
    n_batch: int = 512  # Batch size for prompt processing
    n_threads: Optional[int] = None  # CPU threads (None = auto)
    n_threads_batch: Optional[int] = None  # Threads for batch processing

    # GPU acceleration
    n_gpu_layers: int = DEFAULT_GPU_LAYERS  # -1 = all
    main_gpu: int = 0
    tensor_split: Optional[List[float]] = None

    # Memory and performance
    use_mmap: bool = True
    use_mlock: bool = False
    numa: bool = False
    
    # Generation (can be overridden by HyperparameterConfig)
    logits_all: bool = False # Needed by some features, but uses more memory
    embedding: bool = False

    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        if not self.model_path or not Path(self.model_path).exists():
            errors.append(f"Model path does not exist: {self.model_path}")
        if self.n_ctx < 128:
            errors.append(f"Context size too small: {self.n_ctx}")
        return errors


# --- GGUF Backend Implementation ---

class LlamaCppBackend(LLMBackend):
    """Backend for running models via llama.cpp"""

    def __init__(self,
                 config: GGUFConfig,
                 hyperparams: Optional[HyperparameterConfig] = None):
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed.")

        errors = config.validate()
        if errors:
            raise ConfigurationError(f"Invalid GGUFConfig: {'; '.join(errors)}")

        super().__init__(
            provider_id="local_gguf",
            model_specifier=str(config.model_path),
            hyperparams=hyperparams
        )
        self.config = config
        # self.model_handle is inherited from LLMBackend and will hold the Llama object

    def load(self, verbose: bool = True, **kwargs) -> bool:
        """Load GGUF model."""
        if self.model_handle is not None:
            self.logger.info("GGUF model already loaded.")
            return True

        self.logger.info(f"Loading GGUF model from: {self.config.model_path}")
        if verbose:
            ConsoleOutput.info(f"Loading GGUF model: {self.config.model_path.name}")

        try:
            self.model_handle = Llama(
                model_path=str(self.config.model_path),
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_threads=self.config.n_threads,
                n_threads_batch=self.config.n_threads_batch,
                n_gpu_layers=self.config.n_gpu_layers,
                main_gpu=self.config.main_gpu,
                tensor_split=self.config.tensor_split,
                use_mmap=self.config.use_mmap,
                use_mlock=self.config.use_mlock,
                numa=self.config.numa,
                logits_all=self.config.logits_all,
                embedding=self.config.embedding,
                verbose=verbose  # Pass verbosity setting
            )
            self.logger.info("GGUF model loaded successfully")
            if verbose:
                 ConsoleOutput.success("Model loaded")
                 self._print_model_info()
            return True
        except Exception as e:
            self.model_handle = None # Ensure it's None on failure
            self.logger.error(f"Failed to load GGUF model: {e}", exc_info=True)
            ConsoleOutput.error(f"Failed to load GGUF model: {e}")
            raise ModelLoadError(f"Failed to load GGUF model: {e}", self.model_specifier) from e

    def unload(self):
        """Unload GGUF model and free memory."""
        self.logger.info(f"Unloading GGUF model: {self.model_specifier}")
        model_ref = self.model_handle
        self.model_handle = None
        # llama-cpp doesn't have an explicit unload, rely on Python's GC
        # We use the helper to be thorough
        cleanup_resources([model_ref], clear_cuda=True)
        self.logger.info("GGUF model unloaded.")

    def _print_model_info(self):
        """Print basic model information after loading."""
        if self.model_handle is None: return
        try:
             print(f"  Context size (n_ctx): {self.model_handle.context_params.n_ctx}")
             print(f"  Batch size (n_batch): {self.model_handle.context_params.n_batch}")
             print(f"  GPU Layers (n_gpu_layers): {self.model_handle.n_gpu_layers()}")
             print(f"  CPU Threads: {self.model_handle.context_params.n_threads or 'auto'}")
        except Exception as e:
             self.logger.warning(f"Could not retrieve full model info from llama-cpp: {e}")


    def _generate_request(self,
                          prompt: str,
                          hyperparams: HyperparameterConfig,
                          **kwargs) -> GenerationResult:
        """Internal generation logic for llama.cpp"""

        # Get stop tokens from kwargs if provided (e.g., by chat template)
        stop = kwargs.get("stop", [])

        # Map generic hypers to llama-cpp specific ones
        gen_kwargs = HyperparamMapper.get_llama_cpp_config(hyperparams, self.config.n_ctx)
        
        # Add/override stop tokens
        if stop:
            gen_kwargs["stop"] = list(set(gen_kwargs.get("stop", []) + stop)) # Combine lists, ensure unique

        self.logger.debug(f"Generating with llama.cpp params: {gen_kwargs}")
        
        try:
            # Llama object call handles generation
            output = self.model_handle(prompt, **gen_kwargs)
            
            raw_output_text = output["choices"][0]["text"].strip()
            usage = output.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            # Fallback token counting if usage not present
            if input_tokens == 0:
                 input_tokens = len(self.model_handle.tokenize(prompt.encode('utf-8')))
            if output_tokens == 0:
                 output_tokens = len(self.model_handle.tokenize(raw_output_text.encode('utf-8')))

            return GenerationResult(
                raw_output=raw_output_text,
                filtered_output=raw_output_text, # Filtering done in pipeline
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=0.0, # Will be set by base class decorator
                memory_used=0, # Hard to track
                device_map={"provider": "local_gguf", "gpu_layers": self.config.n_gpu_layers}
            )

        except Exception as e:
            self.logger.error(f"GGUF generation failed: {e}", exc_info=True)
            raise GenerationError(f"GGUF generation failed: {e}", self.model_specifier) from e


    def _chat_request(self,
                      system_prompt: str,
                      user_message: str,
                      hyperparams: HyperparameterConfig,
                      **kwargs) -> GenerationResult:
        """
        Process text using llama_cpp's chat handler.
        This respects the model's built-in chat template if available.
        """
        if self.model_handle is None:
            raise RuntimeError("GGUF model not loaded. Call load_model() first.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        # Map generic hypers to llama-cpp specific ones
        gen_kwargs = HyperparamMapper.get_llama_cpp_config(hyperparams, self.config.n_ctx)
        
        self.logger.debug(f"Generating chat with llama.cpp params: {gen_kwargs}")

        try:
            # Use create_chat_completion
            output = self.model_handle.create_chat_completion(
                messages=messages,
                **gen_kwargs
            )
            
            raw_output_text = output["choices"][0]["message"]["content"].strip()
            usage = output.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            # Fallback token counting if usage not present
            if input_tokens == 0:
                 # Re-create prompt as string to count (less accurate but a fallback)
                 # Note: This is imperfect as it doesn't know the *exact* template used internally
                 prompt_str = f"SYSTEM: {system_prompt}\nUSER: {user_message}\nASSISTANT:"
                 input_tokens = len(self.model_handle.tokenize(prompt_str.encode('utf-8')))
            if output_tokens == 0:
                 output_tokens = len(self.model_handle.tokenize(raw_output_text.encode('utf-8')))

            return GenerationResult(
                raw_output=raw_output_text,
                filtered_output=raw_output_text, # Filtering done in pipeline
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=0.0, # Will be set by base class decorator
                memory_used=0,
                device_map={"provider": "local_gguf", "gpu_layers": self.config.n_gpu_layers}
            )

        except Exception as e:
            self.logger.error(f"GGUF chat generation failed: {e}", exc_info=True)
            raise GenerationError(f"GGUF chat generation failed: {e}", self.model_specifier) from e
