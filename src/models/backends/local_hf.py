# llamanote/models/backends/local_hf.py
"""
Local Hugging Face Transformers Backend
Implements the LLMBackend interface for running models locally using 'transformers'.
"""

import gc
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from ...utils.helpers import DynamicTokenLimitCalculator
import time

# Third-party imports
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig
    )
    # from accelerate import init_empty_weights, infer_auto_device_map # Use "auto" instead
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy types for type hinting if torch/transformers isn't installed
    if not globals().get("torch"):
        class MockTorch:
            float16 = "float16"
            float32 = "float32"
            bfloat16 = "bfloat16"
            @staticmethod
            def cuda(*args, **kwargs): return MockCuda()
        class MockCuda:
             @staticmethod
             def is_available(): return False
             @staticmethod
             def device_count(): return 0
             @staticmethod
             def memory_allocated(*args, **kwargs): return 0
             @staticmethod
             def max_memory_allocated(*args, **kwargs): return 0
             @staticmethod
             def reset_peak_memory_stats(*args, **kwargs): pass
             @staticmethod
             def empty_cache(*args, **kwargs): pass
        torch = MockTorch() # type: ignore
    class AutoModelForCausalLM: pass
    class AutoTokenizer: pass
    class BitsAndBytesConfig: pass
    class GenerationConfig: pass


# Local project imports
from .base import LLMBackend
from ...core.types import GenerationResult, QuantizationConfig, LayerSplitConfig
from ...processing.response_filter import ResponseFilter
from ...core.errors import ModelLoadError, GenerationError
from ...models.hyperparameters import HyperparameterConfig
from ...models.registry import ModelEntry
from ...utils.logger import get_logger_conf, ConsoleOutput
from ...utils.helpers import cleanup_resources, get_device_manager
from ...utils.decorators import log_execution_time, log_resource_usage
from ...config.settings import DEFAULT_CACHE_DIR, DEFAULT_OFFLOAD_DIR

logger = get_logger_conf(__name__)

# --- Model Loader Helper (Refactoring Item 5) ---

class LocalModelLoader:
    """Helper class to consolidate the logic for loading HF Transformers models."""
    
    def __init__(self, model_id: str, quant_config: QuantizationConfig, 
                 split_config: LayerSplitConfig, trust_remote_code: bool):
        self.model_id = model_id
        self.quant_config = quant_config
        self.split_config = split_config
        self.trust_remote_code = trust_remote_code
        self.device_manager = get_device_manager()
        self.load_config = {}

    def build_load_config(self) -> Dict[str, Any]:
        """Builds the kwargs dictionary for AutoModelForCausalLM.from_pretrained."""
        self.load_config = {
            "cache_dir": str(DEFAULT_CACHE_DIR),
            "trust_remote_code": self.trust_remote_code
        }
        
        is_cpu = not self.device_manager.is_cuda_available()

        # 1. Set Device Map & Offloading
        if is_cpu:
            self.load_config["device_map"] = {"": "cpu"}
            self.load_config["low_cpu_mem_usage"] = False
            logger.info("Configuring model for CPU-only load.")
        elif self.split_config.enabled:
            self.load_config["device_map"] = "auto"
            self.load_config["max_memory"] = self.split_config.get_max_memory_dict()
            self.load_config["low_cpu_mem_usage"] = True
            offload_dir = self.split_config.offload_folder or DEFAULT_OFFLOAD_DIR
            if self.split_config.offload_state_dict or offload_dir:
                 self.load_config["offload_folder"] = str(offload_dir)
                 self.load_config["offload_state_dict"] = self.split_config.offload_state_dict
            logger.info(f"Configuring model for layer splitting (device_map='auto') with max_memory: {self.load_config['max_memory']}")
        else:
            self.load_config["device_map"] = "auto"
            self.load_config["low_cpu_mem_usage"] = True
            logger.info("Configuring model for 'auto' device mapping (no explicit splitting).")

        # 2. Set Quantization
        bnb_config = self.quant_config.to_bnb_config() if not is_cpu else None
        if bnb_config:
            self.load_config["quantization_config"] = bnb_config
            self.load_config["torch_dtype"] = self.quant_config.compute_dtype
            logger.info(f"Applying {self.quant_config.method} quantization with {self.quant_config.compute_dtype}.")
        elif not is_cpu:
            self.load_config["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info(f"Using default GPU dtype: {self.load_config['torch_dtype']}")
        else:
             self.load_config["torch_dtype"] = torch.float32
             logger.info("Using default CPU dtype: float32")
             
        # 3. Handle potential 16bit method (which just means setting dtype)
        if self.quant_config.method == "16bit" and "quantization_config" not in self.load_config:
            self.load_config["torch_dtype"] = self.quant_config.compute_dtype or torch.float16
            logger.info(f"Using 16-bit precision: {self.load_config['torch_dtype']}")

        return self.load_config

    def load(self) -> Tuple[Any, Any]:
        """Attempts to load the model and tokenizer."""
        if not TORCH_AVAILABLE:
            raise ModelLoadError("PyTorch/Transformers not installed.", self.model_id)

        load_config = self.build_load_config()
        
        # --- Load Tokenizer ---
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(DEFAULT_CACHE_DIR),
                use_fast=True,
                trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {self.model_id}: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load tokenizer: {e}", self.model_id) from e

        # --- Load Model ---
        try:
            logger.info(f"Attempting to load model '{self.model_id}'...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_config
            )
            logger.info(f"Successfully loaded model: {self.model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id} with config {load_config}: {e}", exc_info=True)
            
            # --- Fallback Logic ---
            # If load failed with quantization or splitting, try a simpler config
            if load_config.get("device_map") != {"": "cpu"} and (load_config.get("quantization_config") or load_config.get("max_memory")):
                logger.warning("Falling back to standard 'auto' device map without quantization/limits.")
                fallback_config = {
                    "cache_dir": str(DEFAULT_CACHE_DIR),
                    "trust_remote_code": self.trust_remote_code,
                    "torch_dtype": torch.bfloat16 if self.device_manager.is_cuda_available() else torch.float32,
                    "device_map": "auto",
                    "low_cpu_mem_usage": self.device_manager.is_cuda_available(),
                }
                try:
                    model = AutoModelForCausalLM.from_pretrained(self.model_id, **fallback_config)
                    logger.info("Successfully loaded model with standard 'auto' fallback.")
                    return model, tokenizer
                except Exception as e2:
                    logger.error(f"Standard 'auto' fallback failed: {e2}", exc_info=True)

            # If that also failed (or wasn't applicable), try CPU
            if load_config.get("device_map") != {"": "cpu"}:
                 logger.warning("Falling back to CPU-only load.")
                 cpu_config = {
                    "cache_dir": str(DEFAULT_CACHE_DIR),
                    "trust_remote_code": self.trust_remote_code,
                    "torch_dtype": torch.float32,
                    "device_map": {"": "cpu"},
                 }
                 try:
                      model = AutoModelForCausalLM.from_pretrained(self.model_id, **cpu_config)
                      logger.info("Successfully loaded model with CPU-only fallback.")
                      return model, tokenizer
                 except Exception as e3:
                      logger.error(f"CPU-only fallback failed: {e3}", exc_info=True)
                      
            # All attempts failed
            raise ModelLoadError(f"All loading attempts failed. Last error: {e}", self.model_id) from e

# --- Main Backend Class ---

class LocalHFBackend(LLMBackend):
    """Implementation for local Hugging Face Transformers models."""

    def __init__(self,
                 model_id: str,
                 model_entry: ModelEntry,
                 quant_config: QuantizationConfig,
                 split_config: LayerSplitConfig,
                 hyperparams: Optional[HyperparameterConfig] = None):
        
        if not TORCH_AVAILABLE:
            raise ImportError("LocalHFBackend requires 'torch' and 'transformers'.")

        super().__init__("local_hf", model_id, hyperparams)
        self.model_id = model_id  # Store for compatibility with helper classes
        self.model_entry = model_entry
        self.quant_config = quant_config
        self.split_config = split_config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device_map: Optional[Dict] = None

        # These helpers depend on the model_entry
        self.response_filter = ResponseFilter(self.model_entry)
        self.token_calculator = DynamicTokenLimitCalculator(
            model_config=self.model_entry
        )
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        self.logger.info(f"Initialized LocalHFBackend for {self.model_id}")

    @log_execution_time(logger_name=__name__)
    @log_resource_usage(logger_name=__name__)
    def load(self, trust_remote_code: bool = True, **kwargs) -> bool:
        """Loads the model and tokenizer using the ModelLoader helper."""
        if self.is_loaded and self.model_handle is not None:
            self.logger.info("Model is already loaded.")
            return True
            
        loader = LocalModelLoader(
            self.model_specifier,
            self.quant_config,
            self.split_config,
            trust_remote_code
        )
        
        try:
            model, tokenizer = loader.load()
            self.model_handle = model
            self.tokenizer = tokenizer
            self._post_load_setup()
            self._log_model_info()
            self.is_loaded = True
            return True
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}", exc_info=True)
            ConsoleOutput.error(f"Failed to load model {self.model_specifier}: {e}")
            self.is_loaded = False
            return False

    def _post_load_setup(self):
        """Common setup tasks after model/tokenizer are loaded."""
        if self.tokenizer and self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.debug("Set pad_token to eos_token")
            else:
                self.logger.warning(f"Model {self.model_specifier} has no EOS or PAD token. Adding [PAD].")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model_handle.resize_token_embeddings(len(self.tokenizer))
                self.hyperparams.pad_token_id = self.tokenizer.pad_token_id
        
        # Sync tokenizer token IDs to hyperparams if not already set
        self.hyperparams.pad_token_id = self.hyperparams.pad_token_id or self.tokenizer.pad_token_id
        # Handle list vs single ID for eos_token_id
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list): eos_id = eos_id[0]
        self.hyperparams.eos_token_id = self.hyperparams.eos_token_id or eos_id
        self.hyperparams.bos_token_id = self.hyperparams.bos_token_id or self.tokenizer.bos_token_id

        # Update device_map info
        if hasattr(self.model_handle, 'hf_device_map'):
             self.device_map = self.model_handle.hf_device_map
        elif self.device == "cpu":
             self.device_map = {"": "cpu"}
        
        self.logger.debug("Post-load setup complete.")


    def _log_model_info(self):
        """Log information about loaded model"""
        if self.model_handle is None: return
        try:
            param_count = sum(p.numel() for p in self.model_handle.parameters())
            param_dtype = next(self.model_handle.parameters()).dtype
            
            bytes_per_param = 4 # float32
            if self.quant_config.method == "4bit": bytes_per_param = 0.5
            elif self.quant_config.method == "8bit": bytes_per_param = 1
            elif param_dtype in [torch.float16, torch.bfloat16]: bytes_per_param = 2
            
            param_size_gb = (param_count * bytes_per_param) / (1024 ** 3)
            self.logger.info(f"Model parameters: {param_count:,} (~{param_size_gb:.2f} GB effective size)")
        except Exception:
            pass # Fails if model has no parameters

        if self.device_map:
            self.logger.info(f"Model Device Map: {self.device_map}")
        
        if self.device_manager.is_cuda_available():
            for i in range(self.device_manager.get_gpu_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    logger.info(f"  GPU {i} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                except Exception: pass # Ignore errors if memory check fails


    def _generate_request(self, 
                          prompt: str, 
                          hyperparams: HyperparameterConfig, 
                          **kwargs) -> GenerationResult:
        """HF-specific generation logic."""
        
        # Calculate dynamic max_new_tokens
        calculated_max = self.token_calculator.calculate_max_new_tokens(prompt)
        max_new = min(hyperparams.max_new_tokens, calculated_max) if hyperparams.max_new_tokens else calculated_max
        
        max_input_length = self.model_entry.max_context - max_new - 10 # Safety buffer

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}", exc_info=True)
            raise GenerationError(f"Tokenization failed: {e}", self.model_id) from e
            
        input_tokens = inputs['input_ids'].shape[1]

        # Move inputs to the model's primary device
        try:
            first_device = next(self.model_handle.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        except Exception as e:
            logger.error(f"Failed to move inputs to device {first_device}: {e}", exc_info=True)
            raise GenerationError(f"Device placement error: {e}", self.model_id) from e

        # Create GenerationConfig dictionary from hyperparams
        gen_config_dict = hyperparams.to_dict()
        gen_config_dict['max_new_tokens'] = max_new
        # Ensure critical token IDs are set from tokenizer if not provided
        gen_config_dict.setdefault('pad_token_id', self.tokenizer.pad_token_id)
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list): eos_id = eos_id[0]
        gen_config_dict.setdefault('eos_token_id', eos_id)

        try:
            gen_config = GenerationConfig(**gen_config_dict)
        except Exception as e:
             logger.error(f"Failed to create GenerationConfig: {e}. Using default.", exc_info=True)
             gen_config = GenerationConfig(max_new_tokens=max_new)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model_handle.generate(
                **inputs,
                generation_config=gen_config
            )
        gen_time = time.time() - start_time
        
        # Decode output
        # Get only the newly generated tokens
        output_tokens_tensor = outputs[0][input_tokens:]
        raw_output = self.tokenizer.decode(output_tokens_tensor, skip_special_tokens=True).strip()
        output_tokens = output_tokens_tensor.shape[0]

        # Filter if requested by kwargs
        remove_thinking = kwargs.get('remove_thinking', True)
        if remove_thinking and self.model_entry.supports_thinking:
            filter_result = self.response_filter.filter(raw_output)
            filtered_output = filter_result.filtered_text
        else:
            filtered_output = raw_output

        # Get memory usage
        memory_used = 0
        if self.device_manager.is_cuda_available():
            try:
                memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Peak MB
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                memory_used = torch.cuda.memory_allocated() / (1024 * 1024) # Current MB

        return GenerationResult(
            raw_output=raw_output,
            filtered_output=filtered_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time=gen_time,
            memory_used=memory_used,
            device_map=self.device_map
        )


    def _chat_request(self, 
                      system_prompt: str, 
                      user_message: str, 
                      hyperparams: HyperparameterConfig, 
                      **kwargs) -> GenerationResult:
        """HF specific chat generation."""
        conversation = []
        if system_prompt:
             conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_message})

        prompt = ""
        try:
            # Check for both attribute and non-empty template string
            if hasattr(self.tokenizer, 'apply_chat_template') and getattr(self.tokenizer, 'chat_template', None):
                prompt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True # Crucial for inference
                )
                self.logger.debug("Applied tokenizer's chat template.")
            else:
                 self.logger.warning(f"Tokenizer {self.model_id} has no chat template. Using basic system/user/assistant format.")
                 prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_message}\n\nASSISTANT:"
        except Exception as e:
            self.logger.warning(f"Chat template application failed: {e}. Using fallback.", exc_info=True)
            prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_message}\n\nASSISTANT:"

        # Delegate to the standard generate method
        return self._generate_request(prompt, hyperparams, **kwargs)

    def unload(self):
        """Unload model and tokenizer, clear memory."""
        if not self.is_loaded:
            return
        self.logger.info(f"Unloading local HF model: {self.model_specifier}")
        cleanup_resources([self.model_handle, self.tokenizer], clear_cuda=True)
        self.model_handle = None
        self.tokenizer = None
        self.device_map = None
        self.is_loaded = False
        self.logger.info("Local HF model unloaded.")
