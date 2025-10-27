"""
llama.cpp Backend Integration
Provides support for GGUF/GGML quantized models via llama-cpp-python
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time

try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    Llama = None
    LlamaGrammar = None

from loggerConf import get_logger_conf, ConsoleOutput
from hyperparameters import HyperparameterConfig
from config import CACHE_DIR

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
    n_gpu_layers: int = 0  # Number of layers to offload to GPU (0 = CPU only)
    main_gpu: int = 0  # Main GPU to use
    tensor_split: Optional[List[float]] = None  # How to split layers across GPUs
    
    # Memory and performance
    use_mmap: bool = True  # Use mmap for faster loading
    use_mlock: bool = False  # Lock model in RAM
    numa: bool = False  # NUMA support
    
    # Model format
    vocab_only: bool = False  # Only load vocabulary
    use_fp16_kv: bool = True  # Use FP16 for key/value cache
    logits_all: bool = False  # Return logits for all tokens
    embedding: bool = False  # Enable embedding mode
    
    # Advanced
    rope_freq_base: float = 10000.0  # RoPE frequency base
    rope_freq_scale: float = 1.0  # RoPE frequency scaling
    mul_mat_q: bool = True  # Use quantized matrix multiplication
    
    # Offloading
    offload_kqv: bool = True  # Offload K, Q, V to GPU
    flash_attn: bool = False  # Use flash attention
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        if self.model_path is None or not Path(self.model_path).exists():
            errors.append(f"Model path does not exist: {self.model_path}")
        
        if self.n_ctx < 128:
            errors.append(f"Context size too small: {self.n_ctx}")
        
        if self.n_gpu_layers < 0:
            errors.append(f"Invalid n_gpu_layers: {self.n_gpu_layers}")
        
        return errors


@dataclass
class LlamaCppGenerationResult:
    """Result from llama.cpp generation"""
    text: str
    tokens: List[int]
    input_tokens: int
    output_tokens: int
    generation_time: float
    tokens_per_second: float
    stop_reason: str  # "length", "eos", "stop_word"


class LlamaCppBackend:
    """Backend for running models via llama.cpp"""
    
    def __init__(self, config: LlamaCppConfig):
        """
        Initialize llama.cpp backend
        
        Args:
            config: LlamaCpp configuration
        """
        if not LLAMACPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )
        
        self.config = config
        self.model: Optional[Llama] = None
        
        # Validate config
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        
        logger.info("Initialized LlamaCppBackend")
    
    def load_model(self, verbose: bool = True) -> bool:
        """
        Load GGUF model
        
        Args:
            verbose: Whether to show loading progress
            
        Returns:
            True if successful
        """
        logger.info(f"Loading model from: {self.config.model_path}")
        
        if verbose:
            ConsoleOutput.info(f"Loading GGUF model: {self.config.model_path.name}")
        
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
                vocab_only=self.config.vocab_only,
                use_fp16_kv=self.config.use_fp16_kv,
                logits_all=self.config.logits_all,
                embedding=self.config.embedding,
                rope_freq_base=self.config.rope_freq_base,
                rope_freq_scale=self.config.rope_freq_scale,
                mul_mat_q=self.config.mul_mat_q,
                offload_kqv=self.config.offload_kqv,
                flash_attn=self.config.flash_attn,
                verbose=verbose
            )
            
            logger.info("Model loaded successfully")
            
            if verbose:
                ConsoleOutput.success("Model loaded")
                self._print_model_info()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if verbose:
                ConsoleOutput.error(f"Failed to load model: {e}")
            return False
    
    def _print_model_info(self):
        """Print model information"""
        if self.model is None:
            return
        
        print(f"  Context size: {self.config.n_ctx}")
        print(f"  GPU layers: {self.config.n_gpu_layers}")
        print(f"  Threads: {self.config.n_threads or 'auto'}")
        print(f"  Memory mapping: {self.config.use_mmap}")
    
    def generate(self,
                prompt: str,
                hyperparams: Optional[HyperparameterConfig] = None,
                stop: Optional[List[str]] = None,
                stream: bool = False) -> LlamaCppGenerationResult:
        """
        Generate text using llama.cpp
        
        Args:
            prompt: Input prompt
            hyperparams: Generation hyperparameters
            stop: Stop sequences
            stream: Whether to stream output (not implemented yet)
            
        Returns:
            LlamaCppGenerationResult
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use default hyperparams if not provided
        hp = hyperparams or HyperparameterConfig()
        
        # Convert hyperparameters to llama.cpp format
        gen_kwargs = {
            "max_tokens": hp.max_new_tokens or 2048,
            "temperature": hp.temperature,
            "top_p": hp.top_p,
            "top_k": hp.top_k,
            "repeat_penalty": hp.repetition_penalty,
            "stop": stop or [],
        }
        
        # Add optional parameters
        if hp.min_length > 0:
            gen_kwargs["min_tokens"] = hp.min_length
        
        logger.debug(f"Generating with params: {gen_kwargs}")
        
        # Generate
        start_time = time.time()
        
        try:
            output = self.model(
                prompt,
                **gen_kwargs,
                echo=False  # Don't echo the prompt
            )
            
            generation_time = time.time() - start_time
            
            # Extract results
            text = output["choices"][0]["text"]
            tokens = output["choices"][0].get("tokens", [])
            
            # Determine stop reason
            finish_reason = output["choices"][0].get("finish_reason", "unknown")
            stop_reason_map = {
                "length": "length",
                "stop": "stop_word",
                None: "eos"
            }
            stop_reason = stop_reason_map.get(finish_reason, "unknown")
            
            # Calculate token counts
            input_tokens = len(self.model.tokenize(prompt.encode()))
            output_tokens = len(tokens) if tokens else len(self.model.tokenize(text.encode()))
            
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
            
            logger.info(f"Generated {output_tokens} tokens in {generation_time:.2f}s "
                       f"({tokens_per_second:.1f} tokens/s)")
            
            return LlamaCppGenerationResult(
                text=text,
                tokens=tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second,
                stop_reason=stop_reason
            )
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.tokenize(text.encode())
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.detokenize(tokens).decode()
    
    def get_context_size(self) -> int:
        """Get model context size"""
        return self.config.n_ctx
    
    def reset(self):
        """Reset model state"""
        if self.model is not None:
            self.model.reset()
            logger.debug("Model state reset")
    
    def unload(self):
        """Unload model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Model unloaded")


class GGUFModelManager:
    """Manager for GGUF format models"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or CACHE_DIR / "gguf_models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized GGUFModelManager (cache: {self.cache_dir})")
    
    def list_available_models(self) -> List[Path]:
        """List available GGUF models in cache"""
        patterns = ["*.gguf", "*.ggml", "*.bin"]
        
        models = []
        for pattern in patterns:
            models.extend(self.cache_dir.glob(pattern))
        
        return sorted(models)
    
    def find_model(self, model_name: str) -> Optional[Path]:
        """Find a model by name"""
        available = self.list_available_models()
        
        for model_path in available:
            if model_name.lower() in model_path.name.lower():
                return model_path
        
        return None
    
    def download_model(self, 
                      url: str,
                      filename: Optional[str] = None) -> Optional[Path]:
        """
        Download a GGUF model from URL
        
        Args:
            url: URL to download from
            filename: Save as this filename (default: extract from URL)
            
        Returns:
            Path to downloaded model or None if failed
        """
        import requests
        from tqdm import tqdm
        
        if filename is None:
            filename = url.split('/')[-1]
        
        output_path = self.cache_dir / filename
        
        if output_path.exists():
            logger.info(f"Model already exists: {output_path}")
            return output_path
        
        logger.info(f"Downloading model from: {url}")
        ConsoleOutput.info(f"Downloading: {filename}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded model to: {output_path}")
            ConsoleOutput.success(f"Downloaded: {filename}")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            ConsoleOutput.error(f"Download failed: {e}")
            
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            
            return None
    
    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Get information about a GGUF model"""
        info = {
            "path": model_path,
            "name": model_path.name,
            "size_mb": model_path.stat().st_size / (1024 * 1024),
            "format": model_path.suffix[1:],  # Remove the dot
        }
        
        # Try to extract quantization info from filename
        name_lower = model_path.name.lower()
        quant_types = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"]
        
        for quant in quant_types:
            if quant in name_lower:
                info["quantization"] = quant.upper()
                break
        
        return info


def create_llamacpp_backend_from_hyperparams(
    model_path: Path,
    hyperparams: HyperparameterConfig,
    n_gpu_layers: int = 0
) -> LlamaCppBackend:
    """
    Create a llama.cpp backend configured from hyperparameters
    
    Args:
        model_path: Path to GGUF model
        hyperparams: Hyperparameter configuration
        n_gpu_layers: Number of layers to offload to GPU
        
    Returns:
        Configured LlamaCppBackend
    """
    config = LlamaCppConfig(
        model_path=model_path,
        n_ctx=hyperparams.max_new_tokens or 2048,
        n_gpu_layers=n_gpu_layers,
        use_mmap=True,
        use_mlock=False,
    )
    
    backend = LlamaCppBackend(config)
    return backend
