"""
llama.cpp Backend Integration
Provides support for GGUF/GGML quantized models via llama-cpp-python
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
import sys
import re
import requests
from tqdm import tqdm

# Try importing llama-cpp-python
try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    # Define dummy types for type hinting if import fails
    Llama = type('Llama', (object,), {})
    LlamaGrammar = type('LlamaGrammar', (object,), {})

# --- LlamaNote Modules ---
from loggerConf import get_logger_conf, ConsoleOutput, log_execution_time
from hyperparameters import HyperparameterConfig
from config_base import CACHE_DIR # Use base config for cache path

# Import necessary types from centralized locations
# Assuming GenerationResult is now in pipeline_types
from pipeline_types import GenerationResult
from llm_handler import LLMBackend # Import the abstract base class

logger = get_logger_conf(__name__)


@dataclass
class LlamaCppConfig:
    """Configuration for llama.cpp backend"""
    # Model loading
    model_path: Path = None
    n_ctx: int = 2048  # Context size
    n_batch: int = 512  # Batch size for prompt processing
    n_threads: Optional[int] = None  # CPU threads (None = auto)
    n_threads_batch: Optional[int] = None  # Threads for batch processing

    # GPU acceleration
    n_gpu_layers: int = 0  # Number of layers to offload to GPU (0 = CPU only, -1 = all)
    main_gpu: int = 0  # Main GPU to use
    tensor_split: Optional[List[float]] = None  # How to split layers across GPUs

    # Memory and performance
    use_mmap: bool = True  # Use mmap for faster loading
    use_mlock: bool = False  # Lock model in RAM
    numa: bool = False  # NUMA support

    # Model format / Advanced (less commonly changed)
    # vocab_only: bool = False
    # use_fp16_kv: bool = True
    # logits_all: bool = False
    # embedding: bool = False
    # rope_freq_base: float = 10000.0
    # rope_freq_scale: float = 1.0
    # mul_mat_q: bool = True
    # offload_kqv: bool = True
    # flash_attn: bool = False # Requires specific llama-cpp-python build

    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        if self.model_path is None or not Path(self.model_path).exists():
            errors.append(f"Model path does not exist: {self.model_path}")
        if self.n_ctx < 128:
            errors.append(f"Context size too small: {self.n_ctx}")
        # n_gpu_layers can be -1, 0, or positive
        # Add more checks if needed
        return errors

# This dataclass might become redundant if GenerationResult covers everything needed
# @dataclass
# class LlamaCppGenerationResult:
#     """Result from llama.cpp generation"""
#     text: str
#     tokens: List[int]
#     input_tokens: int
#     output_tokens: int
#     generation_time: float
#     tokens_per_second: float
#     stop_reason: str  # "length", "eos", "stop_word"


class LlamaCppBackend(LLMBackend): # Inherit from LLMBackend
    """Backend for running models via llama.cpp"""

    def __init__(self, config: LlamaCppConfig, hyperparams: Optional[HyperparameterConfig] = None):
        """
        Initialize llama.cpp backend

        Args:
            config: LlamaCpp configuration
            hyperparams: Default hyperparameters for generation
        """
        if not LLAMACPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )

        # Validate config first
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid LlamaCppConfig: {errors}")

        # Call LLMBackend superclass constructor
        # Model specifier here is the path to the GGUF file
        super().__init__(model_specifier=str(config.model_path), hyperparams=hyperparams)

        self.config = config
        self.model: Optional[Llama] = None
        self.logger.info("Initialized LlamaCppBackend")

    @property
    def provider_identifier(self) -> str:
        """Return provider name."""
        return "local_gguf"

    def load_model(self, verbose: bool = True, **kwargs) -> bool: # Added **kwargs for compatibility
        """
        Load GGUF model

        Args:
            verbose: Whether to show loading progress
            **kwargs: Ignored, for interface compatibility

        Returns:
            True if successful
        """
        self.logger.info(f"Loading GGUF model from: {self.config.model_path}")
        if self.model is not None:
             self.logger.warning("Model already loaded. Unload first if reloading is intended.")
             return True

        if verbose:
            ConsoleOutput.info(f"Loading GGUF model: {Path(self.config.model_path).name}")

        try:
            self.model = Llama(
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
                # Pass only essential loading args, generation args go in generate()
                verbose=verbose
            )
            self.logger.info("GGUF model loaded successfully")
            if verbose:
                ConsoleOutput.success("Model loaded")
                self._print_model_info()
            return True

        except Exception as e:
            self.logger.error(f"Failed to load GGUF model: {e}", exc_info=True)
            if verbose:
                ConsoleOutput.error(f"Failed to load GGUF model: {e}")
            self.model = None # Ensure model is None on failure
            return False

    def _print_model_info(self):
        """Print basic model information after loading."""
        if self.model is None: return
        print(f"  Context size (n_ctx): {self.config.n_ctx}")
        print(f"  GPU Layers (n_gpu_layers): {self.config.n_gpu_layers}")
        print(f"  CPU Threads: {self.config.n_threads or 'auto'}")
        print(f"  Memory Mapping (use_mmap): {self.config.use_mmap}")

    #@log_execution_time() # Decorator inherited? or apply here? Apply here for clarity
    def generate(self,
                 prompt: str,
                 hyperparams: Optional[HyperparameterConfig] = None,
                 stop: Optional[List[str]] = None, # Allow stop sequences
                 stream: bool = False, # Add stream capability (optional)
                 **kwargs) -> GenerationResult: # Return unified GenerationResult
        """
        Generate text using llama.cpp

        Args:
            prompt: Input prompt
            hyperparams: Generation hyperparameters (override instance defaults)
            stop: Optional list of stop sequences
            stream: Whether to stream output (yields results incrementally)
            **kwargs: Additional llama_cpp specific args (like grammar)

        Returns:
            GenerationResult
        """
        if self.model is None:
            raise RuntimeError("GGUF model not loaded. Call load_model() first.")

        hp = hyperparams or self.hyperparams # Use provided or instance hyperparams

        # Map HyperparameterConfig to llama_cpp kwargs
        gen_kwargs = {
            "max_tokens": hp.max_new_tokens or self.config.n_ctx // 2, # Default if not set
            "temperature": hp.temperature if hp.do_sample else 0.0, # Temp 0 for greedy
            "top_p": hp.top_p if hp.do_sample else 1.0,
            "top_k": hp.top_k if hp.do_sample else -1, # llama-cpp uses -1 for disabled
            "repeat_penalty": hp.repetition_penalty,
            "stop": stop or [], # Add stop sequences
            # Add other mappings if needed: mirostat, grammar, logit_bias, etc.
            "mirostat_mode": kwargs.get("mirostat_mode", 0),
            "grammar": kwargs.get("grammar"),
            "logit_bias": kwargs.get("logit_bias"),
            "echo": False # Don't echo the prompt in the output text
        }
        # Filter out None values that llama-cpp might not like
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        self.logger.debug(f"Generating with llama.cpp params: {gen_kwargs}")

        start_time = time.time()
        try:
            # Llama object call handles generation
            output = self.model(prompt, **gen_kwargs)

            generation_time = time.time() - start_time

            # Extract results from llama_cpp output format (dict)
            raw_output_text = output["choices"][0]["text"].strip()

            # Token counts are often in the 'usage' field
            input_tokens = output.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = output.get("usage", {}).get("completion_tokens", 0)

            # Fallback if usage info is missing (older llama-cpp?)
            if input_tokens == 0: input_tokens = len(self.model.tokenize(prompt.encode()))
            if output_tokens == 0: output_tokens = len(self.model.tokenize(raw_output_text.encode()))


            if generation_time > 0:
                tokens_per_second = output_tokens / generation_time
                self.logger.info(f"GGUF generated {output_tokens} tokens in {generation_time:.2f}s "
                                 f"({tokens_per_second:.1f} tokens/s)")
            else:
                 tokens_per_second = float('inf')
                 self.logger.info(f"GGUF generated {output_tokens} tokens (time < 0.01s)")


            # Filtering thinking tokens is usually done externally by ResponseFilter
            filtered_output = raw_output_text # Assume no filtering needed here

            return GenerationResult(
                raw_output=raw_output_text,
                filtered_output=filtered_output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=generation_time,
                memory_used=0, # Hard to track accurately for llama.cpp from here
                device_map={"provider": "local_gguf", "gpu_layers": self.config.n_gpu_layers}
            )

        except Exception as e:
            self.logger.error(f"GGUF generation failed: {e}", exc_info=True)
            # Reraise or return an error state? Reraise for now.
            raise

    #@log_execution_time() # Apply decorator
    def process_with_chat_template(self,
                                   system_prompt: str,
                                   user_message: str,
                                   hyperparams: Optional[HyperparameterConfig] = None,
                                   **kwargs) -> GenerationResult:
        """
        Process text using a basic chat format (GGUF models might not have templates).

        Args:
            system_prompt: System instruction
            user_message: User input
            hyperparams: Override hyperparameters
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult
        """
        # Basic formatting suitable for many instruction-tuned GGUF models
        # This might need adjustment based on the specific model's fine-tuning
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>"
        # Alternative:
        # prompt = f"SYSTEM: {system_prompt}\nUSER: {user_message}\nASSISTANT:"

        self.logger.debug(f"Using basic chat prompt format for GGUF model.")

        return self.generate(prompt, hyperparams=hyperparams, **kwargs)

    def unload_model(self):
        """Unload GGUF model and free memory."""
        self.logger.info(f"Unloading GGUF model: {self.config.model_path.name}")
        if self.model is not None:
            # llama_cpp doesn't have an explicit unload, rely on GC
            del self.model
            self.model = None
            # Attempt garbage collection
            import gc
            gc.collect()
            # If using CUDA, try clearing cache (might help indirectly)
            try:
                 import torch
                 if torch.cuda.is_available():
                      torch.cuda.empty_cache()
            except ImportError:
                 pass
            self.logger.info("GGUF model unloaded.")
        else:
            self.logger.info("No GGUF model was loaded.")

    def __del__(self):
        """Ensure model is unloaded when instance is deleted."""
        try:
            self.unload_model()
        except Exception as e:
            # Suppress errors during cleanup
            print(f"Error during LlamaCppBackend cleanup: {e}", file=sys.stderr)


class GGUFModelManager:
    """Manager for finding, downloading (optional), and inspecting GGUF models."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or CACHE_DIR / "gguf_models") # Specific subdir for GGUF
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger_conf(f"{__name__}.GGUFManager")
        self.logger.info(f"Initialized GGUFModelManager (cache: {self.cache_dir})")

    def list_available_models(self) -> List[Path]:
        """List available GGUF/GGML models in the cache directory."""
        patterns = ["*.gguf", "*.ggml"] # Common extensions
        models = []
        for pattern in patterns:
            models.extend(list(self.cache_dir.glob(pattern)))
        # Filter out any non-files just in case
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
            "format": model_path.suffix.lower().replace(".", ""),
        }
        # Try to infer quantization from filename (common patterns)
        name_lower = model_path.name.lower()
        quant_match = re.search(r"[._-](q\d_[0-9kmls]|f(?:16|32))[._-]", name_lower)
        if quant_match:
            info["quantization_inferred"] = quant_match.group(1).upper()

        return info

    @log_execution_time()
    def download_model(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """
        Download a GGUF/GGML model from a direct URL
        Args:
            url: URL to download from (must be a direct link to the model file)
            filename: Optional filename to save as.  If none, derived from URL.

        Returns:
            Path to the downloaded model file, or None if download failed
        """
        if not filename:
            try:
                # Attempt to get filename from URL path
                filename = Path(url.split('/')[-1]).name
                # Basic sanitation
                filename = re.sub(r'[^\w\-._]', '', filename)
                if not filename.lower().endswith('.gguf', '.ggml', '.bin'):
                    # Append default if no valid extension found
                    filename += ".gguf"
            except Exception:
                # Fallback filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"downloaded_model_{timestamp}.gguf"
                self.logger.warning(f"Could not derive filename from URL, using fallback filename '{filename}'")

            output_path = self.cache_dir / filename

            if output_path.exists():
                self.logger.info(f"Model already exists in cache: {output_path}")
                ConsoleOutput.info(f"Model '{filename}' already cacached.")
                return output_path

            self.logger.info(f"Attempting to download model from: {url} ")
            ConsoleOutput.info(f"Downloading '{filename}' to {self.cache_dir}...")

            try:
                # Use stream=True for large files
                with requests.get(url, stream=True, timeout=300) as response:
                    response.raise_for_status() # Check for HTTP errors

                    total_size_in_bytes = int(response.headers.get('content-length', 0))
                    block_size = 1024 * 1024 # 1MB chunks

                    progress_bar = tqdm(total=total_size_in_bytes, unit='1B', unit_scale=True, desc=f"Downloading {filename}")
                    with open(output_path, 'wb') as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)

                    progress_bar.close()

                    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        raise IOError(f"Download incomplete: Expected {total_size_in_bytes}, got {progress_bar.n}")

                self.logger.info(f"Successfully downloaded model to {output_path}")
                ConsoleOutput.success(f"Model '{filename}' downloaded successfully.")
                return output_path

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Download failed: Network error or invalid URL - {e}", exc_info=True)
                ConsoleOutput.error(f"Download failed: {e}")
                # Clean up partial download if it exists
                if output_path.exists():
                    try: output_path.unlink()
                    except OSError: pass
                return None
            except IOError as e:
                self.logger.error(f"Download failed: File writing error or incomplete download: {e}", exc_info=True)
                ConsoleOutput.error(f"Download failed: {e}")
                if output_path.exists():
                    try: output_path.unlink()
                    except OSError: pass
                return None
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during download: {e}", exc_info=True)
                ConsoleOutput.error(f"Download failed: {e}")
                if output_path.exists():
                    try: output_path.unlink()
                    except OSError: pass
                return None

# --- Helper Function (Optional) ---

def create_llamacpp_backend(
    model_path: Path,
    hyperparams: Optional[HyperparameterConfig] = None,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1 # Default to auto/all GPU layers
) -> Optional[LlamaCppBackend]:
    """Convenience function to create a LlamaCppBackend."""
    try:
        config = LlamaCppConfig(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )
        backend = LlamaCppBackend(config, hyperparams=hyperparams)
        return backend
    except (ValueError, ImportError) as e:
        logger.error(f"Failed to create LlamaCppBackend: {e}")
        return None
