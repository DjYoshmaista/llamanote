"""
LLM Handler Module - Enhanced Version
Manages model loading, optimization, and response generation with advanced quantization and layer splitting
Supports local (Transformers, GGUF) and cloud (OpenAI, Google) backends.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
import gc
import os
import sys
from pathlib import Path
import time
import abc
import httpx # Required for Anthropic proxy setting

# --- LlamaNote Modules ---
from loggerConf import get_logger_conf, log_execution_time, log_resource_usage, ConsoleOutput, LoggingProgress
from config import ModelConfig, MemoryConfig # Keep ModelConfig for local filter/calc, MemoryConfig legacy
from config_base import (
    FALLBACK_MODEL,
    DEFAULT_MODEL,
    CACHE_DIR,
    OFFLOAD_DIR,
    QUANTIZATION_OPTIONS,
    DEFAULT_QUANTIZATION,
    ENABLE_LAYER_SPLITTING,
    DEFAULT_GPU_LAYERS
)
# Import shared types from the new file
from pipeline_types import (
    QuantizationConfig,
    LayerSplitConfig,
    GenerationResult
)
from hyperparameters import HyperparameterConfig
from response_filter import ResponseFilter, DynamicTokenLimitCalculator
from model_registry import get_model_config

# --- Backend-specific Imports (Conditional) ---
try:
    from llama_cpp import Llama, LlamaGrammar
    from llamacpp_backend import LlamaCppBackend, LlamaCppConfig, LlamaCppGenerationResult # Import from its file
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    # Create dummy types for type hinting if not available
    Llama = None
    LlamaGrammar = None
    LlamaCppBackend = None # Keep this name distinct
    LlamaCppConfig = None
    LlamaCppGenerationResult = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_AI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

logger = get_logger_conf(__name__)

def _get_default_hyperparms():
    """Lazy load default hyperparameters"""
    return HyperparameterConfig()

# --- Backend Interface ---
class LLMBackend(abc.ABC):
    """Abstract base class for LLM backends (local or cloud)."""

    def __init__(self, model_specifier: str, hyperparams: HyperparameterConfig):
        self.model_specifier = model_specifier
        self.hyperparams = hyperparams or _get_default_hyperparms()
        self.logger = get_logger_conf(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def load_model(self, **kwargs) -> bool:
        """Load or initialize the backend (might be no-op for APIs)."""
        pass

    @abc.abstractmethod
    @log_execution_time()
    def generate(self, prompt: str, hyperparams: Optional[HyperparameterConfig] = None, **kwargs) -> GenerationResult:
        """Generate text from a prompt."""
        pass

    @abc.abstractmethod
    @log_execution_time()
    def process_with_chat_template(self, system_prompt: str, user_message: str, hyperparams: Optional[HyperparameterConfig] = None, **kwargs) -> GenerationResult:
        """Generate text using a chat format."""
        pass

    @abc.abstractmethod
    def unload_model(self):
        """Unload or clean up resources (might be no-op for APIs)."""
        pass

    @property
    @abc.abstractmethod
    def provider_identifier(self) -> str:
        """Return provider name (e.g., 'local', 'openai')."""
        pass

    @property
    def model_identifier(self) -> str:
        """Return full model identifier including provider."""
        return f"{self.provider_identifier}:{self.model_specifier}"

# --- Local Hugging Face Transformers Backend ---
class LocalTransformerBackend(LLMBackend):
    """Implementation for local Hugging Face Transformers models."""

    def __init__(self,
                 model_id: str,
                 model_config: ModelConfig, # Needs the specific ModelConfig for filter/calc
                 quantization_config: Optional[QuantizationConfig] = None,
                 layer_split_config: Optional[LayerSplitConfig] = None,
                 hyperparameters: Optional[HyperparameterConfig] = None,
                 memory_config: Optional[MemoryConfig] = None): # memory_config is legacy, prefer split_config

        super().__init__(model_specifier=model_id, hyperparams=hyperparameters)

        self.model_config = model_config # Store the specific config
        self.model_id = model_id # Alias for clarity, already in self.model_specifier
        self.quant_config = quantization_config or QuantizationConfig(method="none")

        # Use layer_split_config if provided, else derive from legacy memory_config
        if layer_split_config:
            self.split_config = layer_split_config
        else:
            self.logger.warning("No LayerSplitConfig provided, creating from legacy MemoryConfig.")
            mem_cfg = memory_config or MemoryConfig()
            max_gpu_mem = {}
            if torch.cuda.is_available():
                 # Check CUDA availability before iterating
                 try:
                     num_gpus = torch.cuda.device_count()
                     for i in range(num_gpus):
                         max_gpu_mem[i] = mem_cfg.max_gpu_memory
                 except Exception as e:
                     self.logger.warning(f"Could not get CUDA device count: {e}. Assuming 0 GPUs for split config.")
                     max_gpu_mem = {}

            self.split_config = LayerSplitConfig(
                enabled=(mem_cfg.use_quantization or torch.cuda.is_available()), # Enable if GPU or quant
                max_gpu_memory=max_gpu_mem,
                max_cpu_memory=mem_cfg.max_cpu_memory,
                offload_folder=OFFLOAD_DIR if mem_cfg.offload_to_disk else None,
                offload_state_dict=mem_cfg.offload_to_disk
            )

        self.model = None
        self.tokenizer = None
        self.device = None
        self.accelerator = None # Accelerator might not be needed with device_map
        self.device_map = None

        self.response_filter = ResponseFilter(self.model_config)
        self.token_calculator = DynamicTokenLimitCalculator(
            self.model_config,
            # Use max_context from the *specific* ModelConfig passed to init
            model_max_context=self.model_config.max_context
        )

        self.logger.info(f"Initialized LocalTransformerBackend for {self.model_id}")
        self.logger.info(f"  Quantization: {self.quant_config.method}")
        self.logger.info(f"  Layer splitting enabled: {self.split_config.enabled}")

    @property
    def provider_identifier(self) -> str:
        return "local"

    @log_execution_time()
    @log_resource_usage()
    def load_model(self, trust_remote_code: bool = True, **kwargs) -> bool:
        """
        Load model with optimizations

        Args:
            trust_remote_code: Whether to trust remote code in model
        """
        self.logger.info(f"Loading model: {self.model_id}") # Use self.model_id

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        success = False # Initialize success flag
        if self.device == "cpu":
            self.logger.warning("No GPU detected, loading on CPU (will be slower)")
            self.split_config.enabled = False # Force no splitting if no CUDA
            self.quant_config.method = "none" # Force no BNB quantization on CPU
            success = self._load_cpu_model(trust_remote_code)
        else:
            try:
                if self.quant_config.method != "none":
                    success = self._load_quantized_model(trust_remote_code)
                elif self.split_config.enabled:
                    success = self._load_split_model(trust_remote_code)
                else:
                    success = self._load_standard_model(trust_remote_code)
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_id}: {e}", exc_info=True)
                self.unload_model() # Ensure cleanup on failure
                return False

        if success:
            self._post_load_setup()
            self._log_model_info()

        return success

    def _load_quantized_model(self, trust_remote_code: bool) -> bool:
        """Load model with quantization"""
        self.logger.info(f"Loading with {self.quant_config.method} quantization")

        bnb_config_dict = self.quant_config.to_bnb_config()
        if bnb_config_dict is None:
            self.logger.warning(f"Quantization method '{self.quant_config.method}' selected but config is None. Loading standard model.")
            return self._load_standard_model(trust_remote_code)

        try:
            # Create BitsAndBytesConfig object here
            bnb_config = BitsAndBytesConfig(**bnb_config_dict)
        except Exception as e:
            self.logger.error(f"Failed to create BitsAndBytesConfig: {e}. Loading standard model.", exc_info=True)
            return self._load_standard_model(trust_remote_code)

        device_map = "auto"
        max_memory = None
        if self.split_config.enabled and torch.cuda.is_available(): # Ensure split config is only used with CUDA
            device_map = self._create_device_map()
            max_memory = self.split_config.get_max_memory_dict()
            self.logger.debug(f"Using device_map='{device_map}' and max_memory={max_memory}")
        elif not torch.cuda.is_available():
            device_map = "cpu" # Force CPU if CUDA not available despite config
            self.logger.warning("CUDA not available, forcing device_map='cpu' for quantized load.")


        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                max_memory=max_memory if device_map != "cpu" else None, # Don't pass max_memory if mapping to CPU
                torch_dtype=self.quant_config.compute_dtype,
                low_cpu_mem_usage=True if device_map != "cpu" else False, # Only use if not fully on CPU
                cache_dir=str(CACHE_DIR), # Ensure string
                offload_folder=str(self.split_config.offload_folder or OFFLOAD_DIR) if self.split_config.enabled and device_map != "cpu" else None,
                offload_state_dict=self.split_config.offload_state_dict if self.split_config.enabled and device_map != "cpu" else False,
                trust_remote_code=trust_remote_code
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(CACHE_DIR),
                use_fast=True,
                trust_remote_code=trust_remote_code
            )

            self.model = model
            self.tokenizer = tokenizer
            if hasattr(model, 'hf_device_map'):
                self.device_map = model.hf_device_map

            self.logger.info(f"Successfully loaded quantized model")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load quantized model: {e}", exc_info=True)
            self.logger.info("Falling back to standard loading")
            return self._load_standard_model(trust_remote_code) # Attempt fallback

    def _load_split_model(self, trust_remote_code: bool) -> bool:
        """Load model with layer splitting (no quantization)"""
        self.logger.info("Loading model with layer splitting")

        if not torch.cuda.is_available():
             self.logger.error("Layer splitting requires CUDA, but it's not available. Falling back.")
             return self._load_cpu_model(trust_remote_code) # Fallback to CPU

        device_map = self._create_device_map()
        max_memory = self.split_config.get_max_memory_dict()

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.bfloat16, # Default to bfloat16
                low_cpu_mem_usage=True,
                cache_dir=str(CACHE_DIR),
                offload_folder=str(self.split_config.offload_folder or OFFLOAD_DIR),
                offload_state_dict=self.split_config.offload_state_dict,
                trust_remote_code=trust_remote_code
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(CACHE_DIR),
                use_fast=True,
                trust_remote_code=trust_remote_code
            )

            if hasattr(model, 'hf_device_map'):
                self.device_map = model.hf_device_map

            self.model = model
            self.tokenizer = tokenizer
            self.logger.info("Successfully loaded model with layer splitting")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load split model: {e}", exc_info=True)
            self.logger.info("Falling back to standard loading")
            return self._load_standard_model(trust_remote_code) # Attempt fallback

    def _load_standard_model(self, trust_remote_code: bool) -> bool:
        """Load model without special optimizations (full precision on GPU/auto)"""
        self.logger.info("Loading model in standard mode (no quantization/splitting)")

        device_map_setting = "auto" if torch.cuda.is_available() else "cpu"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if device_map_setting != "cpu" else torch.float32, # Use bfloat16 on GPU, float32 on CPU
                device_map=device_map_setting,
                cache_dir=str(CACHE_DIR),
                low_cpu_mem_usage=True if device_map_setting != "cpu" else False,
                trust_remote_code=trust_remote_code
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(CACHE_DIR),
                use_fast=True,
                trust_remote_code=trust_remote_code
            )

            self.model = model
            self.tokenizer = tokenizer
            if hasattr(model, 'hf_device_map'):
                self.device_map = model.hf_device_map
            elif device_map_setting == "cpu":
                self.device_map = {"model": "cpu"} # Explicitly set for CPU load

            self.logger.info(f"Successfully loaded model in standard mode on {device_map_setting}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load standard model: {e}", exc_info=True)
            # Try CPU fallback if standard GPU/auto failed
            if device_map_setting != "cpu":
                self.logger.info("Attempting fallback to CPU loading...")
                return self._load_cpu_model(trust_remote_code)
            return False # Final fallback failed

    def _load_cpu_model(self, trust_remote_code: bool) -> bool:
        """Load model on CPU only"""
        self.logger.info("Loading model explicitly on CPU")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32, # CPU usually prefers float32
                # Explicitly avoid device_map='cpu' which can cause issues with from_pretrained
                # Let transformers handle placing it on CPU by default when no CUDA visible
                cache_dir=str(CACHE_DIR),
                low_cpu_mem_usage=False,
                trust_remote_code=trust_remote_code
            ).to('cpu') # Ensure it's moved to CPU if loaded elsewhere initially

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(CACHE_DIR),
                use_fast=True,
                trust_remote_code=trust_remote_code
            )

            self.model = model
            self.tokenizer = tokenizer
            self.device_map = {"": "cpu"} # Set map manually for CPU
            self.device = "cpu" # Update internal device state

            self.logger.info("Successfully loaded model on CPU")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load CPU model: {e}", exc_info=True)
            return False

    def _post_load_setup(self):
        """Common setup tasks after model/tokenizer are loaded."""
        if self.tokenizer and self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.debug("Set pad_token to eos_token")
            else:
                # Default pad token id for models without eos/pad
                default_pad_id = 0
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer)) # Resize model embeddings
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
                if self.tokenizer.pad_token_id is None: # Fallback if adding fails somehow
                    self.tokenizer.pad_token_id = default_pad_id
                self.logger.warning(f"Tokenizer missing pad_token and eos_token. Added '[PAD]' token with id {self.tokenizer.pad_token_id}.")


        # Update hyperparameters with tokenizer info if not already set
        if self.tokenizer:
            # Check for None before assigning, respect existing values if set
            if self.hyperparams.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                self.hyperparams.pad_token_id = self.tokenizer.pad_token_id
            if self.hyperparams.eos_token_id is None and self.tokenizer.eos_token_id is not None:
                # Handle cases where eos_token_id might be a list
                if isinstance(self.tokenizer.eos_token_id, list):
                     self.hyperparams.eos_token_id = self.tokenizer.eos_token_id[0] # Take the first one
                else:
                     self.hyperparams.eos_token_id = self.tokenizer.eos_token_id
            if self.hyperparams.bos_token_id is None and self.tokenizer.bos_token_id is not None:
                self.hyperparams.bos_token_id = self.tokenizer.bos_token_id

        # Update token calculator with model config's max context (using the specific config)
        self.token_calculator.model_max_context = self.model_config.max_context

        # Update response filter with model config (using the specific config)
        self.response_filter.model_config = self.model_config

    def _create_device_map(self) -> Dict[str, Any]:
        """Create optimal device map for layer splitting"""
        if not self.split_config.enabled or not torch.cuda.is_available():
            return "auto" # Let transformers handle CPU if CUDA not available

        if self.split_config.gpu_layers == 0:
            return {"": "cpu"}

        # For transformers, 'auto' with max_memory constraints is generally preferred
        # over trying to manually count layers, which is complex.
        # n_gpu_layers is more relevant for llama.cpp

        if self.split_config.gpu_layers > 0:
             # Accelerate's infer_auto_device_map can sometimes use n_gpu_layers hint,
             # but max_memory is the primary driver. We log a warning.
             self.logger.warning("Manual 'gpu_layers' for Transformers backend is less reliable; "
                               "using 'auto' with 'max_gpu_memory' constraint is preferred.")

        return "auto" # Let accelerate handle it based on max_memory

    def _get_max_memory_dict(self) -> Dict:
        """Get max memory dictionary for device mapping"""
        return self.split_config.get_max_memory_dict()

    def _log_model_info(self):
        """Log information about loaded model"""
        if self.model is None:
            return

        try:
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad) # Count trainable params
            # Estimate size based on effective precision
            param_dtype = next(self.model.parameters()).dtype
            bytes_per_param = 4 # Default float32
            if self.quant_config.method == "4bit":
                bytes_per_param = 0.5
            elif self.quant_config.method == "8bit":
                 bytes_per_param = 1
            elif param_dtype in [torch.float16, torch.bfloat16]:
                 bytes_per_param = 2

            param_size_gb = (param_count * bytes_per_param) / (1024 ** 3)
            self.logger.info(f"Model loaded: ~{param_count:,} parameters (~{param_size_gb:.2f} GB effective size)")
        except Exception as e:
             self.logger.warning(f"Could not calculate model parameter count/size: {e}")

        # Log device map if available
        if self.device_map:
            self.logger.info(f"Device map: {self.device_map}")
            # More detailed breakdown
            device_counts = {}
            total_modules = 0
            # Iterate through values which represent devices
            for device in self.device_map.values():
                 device_str = str(device) # Convert device object/index to string
                 device_counts[device_str] = device_counts.get(device_str, 0) + 1
                 total_modules += 1 # Count modules based on map entries

            if total_modules > 0: # Check if map is not empty
                 for device, count in device_counts.items():
                     self.logger.info(f"  {device}: {count} modules assigned")
                 self.logger.info(f"  (Total modules in device map: {total_modules})")
            else:
                 self.logger.info("  Device map is empty or not detailed.")


        # Log memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    logger.info(f"  GPU {i} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                except Exception as e:
                    logger.warning(f"Could not get memory info for GPU {i}: {e}")

    @log_execution_time()
    def generate(self,
                prompt: str,
                hyperparams: Optional[HyperparameterConfig] = None,
                remove_thinking: bool = True,
                **kwargs) -> GenerationResult:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            hyperparams: Override hyperparameters (uses instance config if None)
            remove_thinking: Whether to filter thinking tokens

        Returns:
            GenerationResult with generated text and metadata
        """
        if self.model is None or self.tokenizer is None:
            self.logger.error("Attempted to generate text but model or tokenizer is not loaded.")
            raise RuntimeError("Model not loaded. Call load_model() first.")

        hp = hyperparams or self.hyperparams

        # Calculate max_new_tokens dynamically if needed
        calculated_max_new_tokens = self.token_calculator.calculate_max_new_tokens(
            input_text=prompt
            # model_max_context is already set in the calculator instance
        )

        if hp.max_new_tokens is None:
            final_max_new_tokens = calculated_max_new_tokens
        else:
            final_max_new_tokens = min(hp.max_new_tokens, calculated_max_new_tokens)


        self.logger.debug(f"Generating with max_new_tokens={final_max_new_tokens}, "
                    f"temperature={hp.temperature}, top_p={hp.top_p}")

        # Tokenize input, ensure truncation respects the calculated output length
        max_input_length = self.model_config.max_context - final_max_new_tokens - 10 # Add safety buffer
        if max_input_length <= 0:
             self.logger.error(f"Calculated max input length is too small ({max_input_length}). "
                               f"Model context {self.model_config.max_context}, requested new tokens {final_max_new_tokens}. Prompt might be too long.")
             # Handle error gracefully - maybe return empty result or raise specific error
             return GenerationResult(raw_output="Error: Prompt too long for model context.", filtered_output="Error: Prompt too long.",
                                     input_tokens=0, output_tokens=0, generation_time=0, memory_used=0)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=max_input_length)
        input_tokens = inputs['input_ids'].shape[1]

        # Move to correct device
        first_device = self._get_model_device()
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        # Create generation config from hyperparameters, using the final max_new_tokens
        gen_config_dict = hp.to_dict()
        gen_config_dict['max_new_tokens'] = final_max_new_tokens
        
        # Ensure essential token IDs are present, use tokenizer's if available
        if 'pad_token_id' not in gen_config_dict and self.tokenizer.pad_token_id is not None:
             gen_config_dict['pad_token_id'] = self.tokenizer.pad_token_id
        # Handle cases where EOS might be a list
        eos_token_id_to_use = self.tokenizer.eos_token_id
        if isinstance(eos_token_id_to_use, list):
            eos_token_id_to_use = eos_token_id_to_use[0] # Use the first one
        if 'eos_token_id' not in gen_config_dict and eos_token_id_to_use is not None:
             gen_config_dict['eos_token_id'] = eos_token_id_to_use
             
        # Ensure pad_token_id and eos_token_id are not None before creating GenerationConfig
        # Fallback if still None after checking tokenizer (should be rare after post_load_setup)
        if gen_config_dict.get('pad_token_id') is None:
            gen_config_dict['pad_token_id'] = 0 # Arbitrary fallback
            self.logger.warning("pad_token_id is None, falling back to 0.")
        if gen_config_dict.get('eos_token_id') is None:
             # Need a valid EOS, trying pad token as last resort
             gen_config_dict['eos_token_id'] = gen_config_dict['pad_token_id']
             self.logger.warning("eos_token_id is None, falling back to pad_token_id.")


        gen_config = GenerationConfig(**gen_config_dict)

        # Generate
        start_time = time.time()
        output_tokens = 0 # Initialize
        raw_output = "" # Initialize
        full_output = "" # Initialize

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )

            generation_time = time.time() - start_time

            # Decode output
            full_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract just the generated portion
            # Find the prompt in the output (it might not be an exact prefix)
            prompt_in_output_idx = full_output.find(prompt)
            if prompt_in_output_idx == 0: # Exact prefix match
                 raw_output = full_output[len(prompt):].strip()
            else:
                 # Fallback: decode only the new tokens
                 # Ensure output tensor has more tokens than input
                 if outputs.shape[1] > input_tokens:
                      output_tokens_tensor = outputs[0][input_tokens:]
                      raw_output = self.tokenizer.decode(output_tokens_tensor, skip_special_tokens=True).strip()
                 else:
                      # Handle case where no new tokens were generated
                      raw_output = ""
                      self.logger.warning("Generation resulted in no new tokens.")

            output_tokens = outputs.shape[1] - input_tokens if outputs.shape[1] > input_tokens else 0


        except Exception as gen_err:
             generation_time = time.time() - start_time
             self.logger.error(f"Generation failed after {generation_time:.2f}s: {gen_err}", exc_info=True)
             raw_output = f"[Generation Error: {gen_err}]"
             # Try to estimate input tokens if possible
             try: input_tokens = self.tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
             except: input_tokens = 0
             output_tokens = 0


        # Filter if requested
        if remove_thinking and self.model_config.supports_thinking:
            filter_result = self.response_filter.filter(raw_output) # Use the instance filter
            filtered_output = filter_result.filtered_text
            self.logger.debug(f"Filtered {filter_result.removal_ratio:.1%} of output")
        else:
            filtered_output = raw_output

        # Get memory usage
        memory_used = 0
        if torch.cuda.is_available():
            try:
                # Use max_memory_allocated for peak usage if available
                memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Peak MB
                torch.cuda.reset_peak_memory_stats() # Reset for next call
            except Exception:
                try:
                    memory_used = torch.cuda.memory_allocated() / (1024 * 1024) # Current MB as fallback
                except Exception:
                    pass # Ignore if fails

        if generation_time > 0 and output_tokens > 0:
            self.logger.info(f"Generated {output_tokens} tokens in {generation_time:.2f}s "
                        f"({output_tokens/generation_time:.1f} tokens/s)")
        elif output_tokens > 0:
            self.logger.info(f"Generated {output_tokens} tokens (generation time < 0.01s)")
        else:
             self.logger.warning("Generation produced 0 output tokens.")

        return GenerationResult(
            raw_output=raw_output,
            filtered_output=filtered_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time=generation_time,
            memory_used=memory_used,
            device_map=self.device_map
        )

    def _get_model_device(self) -> torch.device:
        """Get the device of the first model parameter or the assigned device."""
        if self.model is None:
            # Fallback to CPU if model isn't loaded
            return torch.device("cpu")

        try:
            # This is the most reliable way for models with device_map
            return next(self.model.parameters()).device
        except StopIteration:
            # Model might have no parameters, or hasn't been moved yet
            pass
        except Exception as e:
            self.logger.warning(f"Could not determine model device from parameters: {e}")

        # Fallback to the device determined during load_model
        if self.device:
            return torch.device(self.device)

        # Ultimate fallback
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @log_execution_time()
    def process_with_chat_template(self,
                                  system_prompt: str,
                                  user_message: str,
                                  hyperparams: Optional[HyperparameterConfig] = None,
                                  **kwargs) -> GenerationResult:
        """
        Process text using chat template

        Args:
            system_prompt: System instruction
            user_message: User input
            hyperparams: Override hyperparameters
            **kwargs: Additional arguments (like remove_thinking)

        Returns:
            GenerationResult
        """
        if self.model is None or self.tokenizer is None:
            self.logger.error("Attempted to generate (chat) but model or tokenizer is not loaded.")
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Format as chat
        conversation = []
        if system_prompt: # Only add system prompt if provided
             conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_message})

        # Try to use chat template
        prompt = ""
        try:
            # Check for both attribute and non-empty template string
            if hasattr(self.tokenizer, 'apply_chat_template') and getattr(self.tokenizer, 'chat_template', None):
                prompt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True # Important for inference
                )
                self.logger.debug("Applied tokenizer's chat template.")
            else:
                 # Fallback formatting if no template available
                 self.logger.warning(f"Model {self.model_id} tokenizer has no chat template or it's empty. Using basic formatting.")
                 prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_message}\n\nASSISTANT:"
                 # For some models, just user/assistant might be better:
                 # prompt = f"USER: {user_message}\nASSISTANT:"

        except Exception as e:
            self.logger.warning(f"Chat template application failed: {e}. Using fallback formatting.", exc_info=True)
            prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_message}\n\nASSISTANT:"

        # Pass remove_thinking flag from kwargs or default to True
        remove_thinking = kwargs.get('remove_thinking', True)

        return self.generate(prompt, hyperparams=hyperparams, remove_thinking=remove_thinking)

    def unload_model(self):
        """Unload model and free memory"""
        self.logger.info(f"Unloading model: {self.model_id}")

        # Explicitly delete model and tokenizer references
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if hasattr(self, 'accelerator') and self.accelerator is not None:
             del self.accelerator
             self.accelerator = None

        self.device_map = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                self.logger.debug("CUDA cache cleared.")
            except Exception as e:
                self.logger.warning(f"Could not clear CUDA cache: {e}")

        # Run garbage collection forcefully
        collected = gc.collect()
        self.logger.debug(f"Garbage collector ran, collected {collected} objects.")

        self.logger.info("Model unloaded and memory freed")

    def __del__(self):
        """Ensure cleanup on object deletion"""
        try:
            self.unload_model()
        except Exception as e:
            # Log error during cleanup, but don't raise exception
            print(f"Error during LocalTransformerBackend cleanup: {e}", file=sys.stderr)
            # Use logger if available, otherwise print
            try: self.logger.error(f"Error during __del__: {e}", exc_info=False)
            except: pass


# --- OpenAI Cloud Backend ---
class OpenAIBackend(LLMBackend):
    """Implementation for OpenAI API."""

    def __init__(self, api_key: str, model_specifier: str = "gpt-4o", hyperparameters: Optional[HyperparameterConfig] = None):
        super().__init__(model_specifier, hyperparameters)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run 'pip install openai'")

        self.api_key = api_key
        self.client = None # Initialize in load_model

    @property
    def provider_identifier(self) -> str:
        return "openai"

    def load_model(self, **kwargs) -> bool:
        """Initialize the OpenAI client and test the API key."""
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.client.models.list() # Test API key by listing models
            self.logger.info("OpenAI API client initialized and key verified.")
            return True
        except openai.AuthenticationError:
             self.logger.error("OpenAI API key is invalid.")
             ConsoleOutput.error("OpenAI API key is invalid. Please check your configuration.")
             return False
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client or verify key: {e}", exc_info=True)
            ConsoleOutput.error(f"OpenAI connection failed: {e}. Check key and network access.")
            return False

    def _map_hyperparams(self, hp: HyperparameterConfig) -> Dict[str, Any]:
        """Maps HyperparameterConfig to OpenAI's API parameters."""
        params = {}
        if hp.max_new_tokens is not None:
            params["max_tokens"] = hp.max_new_tokens
        if hp.temperature is not None:
             # Ensure temperature is within valid range (0 to 2)
             params["temperature"] = max(0.0, min(2.0, hp.temperature))
        if hp.top_p is not None:
             # OpenAI API often expects top_p < 1.0 or None, handle 1.0 case
             params["top_p"] = max(0.001, min(0.999, hp.top_p)) if hp.top_p < 1.0 else None
        # Handle do_sample implicitly: OpenAI samples if temperature > 0
        if not hp.do_sample:
            params["temperature"] = 0.0 # Force deterministic output

        # Map repetition_penalty to presence_penalty (different concepts, but closest match)
        if hp.repetition_penalty is not None and hp.repetition_penalty != 1.0:
             # Scale appropriately: 1.0 (HF) -> 0.0 (OpenAI), 1.2 (HF) -> 0.2 (OpenAI)
             # Clamp between -2.0 and 2.0 as per OpenAI docs
             presence_penalty = max(-2.0, min(2.0, hp.repetition_penalty - 1.0))
             # Avoid setting penalty to 0.0 if repetition_penalty was 1.0
             if abs(presence_penalty) > 1e-6 :
                  params["presence_penalty"] = presence_penalty
                  self.logger.debug(f"Mapping repetition_penalty {hp.repetition_penalty} to presence_penalty {params['presence_penalty']}")

        # Map no_repeat_ngram_size to frequency_penalty (also different, heuristic mapping)
        if hp.no_repeat_ngram_size is not None and hp.no_repeat_ngram_size > 0:
            # Higher ngram size implies stronger need to penalize frequency
            # Clamp between -2.0 and 2.0
            frequency_penalty = max(-2.0, min(2.0, 0.1 * hp.no_repeat_ngram_size)) # Small penalty
            if abs(frequency_penalty) > 1e-6:
                params["frequency_penalty"] = frequency_penalty
                self.logger.debug(f"Mapping no_repeat_ngram_size {hp.no_repeat_ngram_size} to frequency_penalty {params['frequency_penalty']}")

        # Note: OpenAI API doesn't support top_k, length_penalty, etc. directly.

        return params

    @log_execution_time()
    def generate(self, prompt: str, hyperparams: Optional[HyperparameterConfig] = None, **kwargs) -> GenerationResult:
        """Generate text using OpenAI API (ChatCompletion with single user prompt)."""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Call load_model() first.")

        hp = hyperparams or self.hyperparams
        api_params = self._map_hyperparams(hp)

        messages = [
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()
        raw_output = ""
        input_tokens = 0
        output_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_specifier,
                messages=messages,
                **api_params
            )
            generation_time = time.time() - start_time

            raw_output = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            memory_used = 0 # N/A

            if generation_time > 0 and output_tokens > 0:
                 tokens_per_sec = output_tokens / generation_time
                 self.logger.info(f"OpenAI generated {output_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")
            elif output_tokens > 0:
                 self.logger.info(f"OpenAI generated {output_tokens} tokens")
            else:
                 self.logger.warning("OpenAI generated 0 tokens.")


        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"OpenAI API call failed after {generation_time:.2f}s: {e}", exc_info=True)
            raw_output = f"[OpenAI Error: {e}]"
            # Try to estimate input tokens
            try: input_tokens = len(prompt.split()) # Very rough estimate
            except: input_tokens = 0
            output_tokens = 0


        # Cloud models generally don't output <think> tags
        filtered_output = raw_output

        return GenerationResult(
            raw_output=raw_output,
            filtered_output=filtered_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time=generation_time,
            memory_used=0, # N/A
            device_map={"provider": "openai"}
        )

    @log_execution_time()
    def process_with_chat_template(self, system_prompt: str, user_message: str, hyperparams: Optional[HyperparameterConfig] = None, **kwargs) -> GenerationResult:
        """Generate text using OpenAI API with system and user prompts."""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Call load_model() first.")

        hp = hyperparams or self.hyperparams
        api_params = self._map_hyperparams(hp)

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        start_time = time.time()
        raw_output = ""
        input_tokens = 0
        output_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_specifier,
                messages=messages,
                **api_params
            )
            generation_time = time.time() - start_time

            raw_output = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            memory_used = 0 # N/A
            filtered_output = raw_output

            if generation_time > 0 and output_tokens > 0:
                 tokens_per_sec = output_tokens / generation_time
                 self.logger.info(f"OpenAI (chat) generated {output_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")
            elif output_tokens > 0:
                 self.logger.info(f"OpenAI (chat) generated {output_tokens} tokens")
            else:
                 self.logger.warning("OpenAI (chat) generated 0 tokens.")

        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"OpenAI API chat call failed after {generation_time:.2f}s: {e}", exc_info=True)
            raw_output = f"[OpenAI Error: {e}]"
            # Estimate tokens roughly
            try: input_tokens = len(system_prompt.split()) + len(user_message.split())
            except: input_tokens = 0
            output_tokens = 0


        return GenerationResult(
            raw_output=raw_output,
            filtered_output=filtered_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time=generation_time,
            memory_used=0, # N/A
            device_map={"provider": "openai"}
        )

    def unload_model(self):
        # No explicit unloading needed, but clear client reference
        if hasattr(self, 'client') and self.client is not None:
             del self.client
             self.client = None
             self.logger.debug("OpenAI client reference cleared.")
        pass

# --- Google Gemini Cloud Backend ---
class GoogleAIBackend(LLMBackend):
    """Implementation for Google AI (Gemini) API."""

    def __init__(self, api_key: str, model_specifier: str, hyperparameters: Optional[HyperparameterConfig] = None):
        super().__init__(model_specifier, hyperparameters)
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Run 'pip install google-generativeai'")

        self.api_key = api_key
        self.model = None # Will be initialized in load_model
        # Store system prompt separately as it's part of model init for Google
        self._system_prompt: Optional[str] = None

    @property
    def provider_identifier(self) -> str:
        return "google"

    def load_model(self, **kwargs) -> bool:
        """Initialize the Google AI client."""
        try:
            genai.configure(api_key=self.api_key)
            # Initialize the base model reference here
            self.model = genai.GenerativeModel(self.model_specifier)
            # Test the model/key with a simple, cheap call (count tokens)
            self.model.count_tokens("test connection")
            self.logger.info(f"Initialized GoogleAIBackend for model: {self.model_specifier} and key verified.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Google AI client or verify key: {e}", exc_info=True)
            # Check for common API key error messages
            err_str = str(e).lower()
            if "api_key_invalid" in err_str or "api key not valid" in err_str or "permission denied" in err_str:
                 ConsoleOutput.error("The provided Google API key is invalid or not enabled for the Generative Language API.")
            else:
                 ConsoleOutput.error(f"Google API connection failed: {e}")
            return False

    def _map_hyperparams(self, hp: HyperparameterConfig) -> Dict[str, Any]:
        """Maps HyperparameterConfig to Google's GenerationConfig."""
        gen_config_params = {}

        # Ensure temperature is set, default to 0.0 for deterministic if sample is False
        if not hp.do_sample:
            gen_config_params["temperature"] = 0.0
        elif hp.temperature is not None:
            # Google API temperature range is typically [0.0, 1.0] (sometimes up to 2.0, check docs)
            gen_config_params["temperature"] = max(0.0, min(1.0, hp.temperature))

        if hp.max_new_tokens is not None:
            gen_config_params["max_output_tokens"] = hp.max_new_tokens

        if hp.top_p is not None:
             # Google API accepts top_p=1.0, range [0.0, 1.0]
             gen_config_params["top_p"] = max(0.0, min(1.0, hp.top_p))

        if hp.top_k is not None and hp.top_k > 0:
             gen_config_params["top_k"] = hp.top_k

        # Map stop sequences if provided
        # if hp.stop_sequences: # Assuming stop_sequences is added to HyperparameterConfig
        #    gen_config_params["stop_sequences"] = hp.stop_sequences

        # Note: Google API doesn't support repetition_penalty, length_penalty, etc.
        if hasattr(hp, 'repetition_penalty') and hp.repetition_penalty != 1.0:
            self.logger.warning("repetition_penalty is not supported by Google Gemini API.")

        return gen_config_params

    @log_execution_time()
    def generate(self, prompt: str, hyperparams: Optional[HyperparameterConfig] = None, **kwargs) -> GenerationResult:
        """Generate text using Google AI (treating raw prompt as user message)."""
        if self.model is None:
            raise RuntimeError("Google AI Model not loaded. Call load_model() first.")

        hp = hyperparams or self.hyperparams
        generation_config_dict = self._map_hyperparams(hp)
        generation_config = genai.types.GenerationConfig(**generation_config_dict)

        # Use the model instance possibly configured with a system prompt
        model_to_use = self.model
        # Note: system_instruction is part of the model object, not generation config

        start_time = time.time()
        raw_output = ""
        input_tokens = 0
        output_tokens = 0
        try:
            # Use generate_content for flexibility (handles multimodal later if needed)
            response = model_to_use.generate_content(
                prompt,
                generation_config=generation_config
            )
            generation_time = time.time() - start_time

            # Handle potential safety blocks or empty responses
            if not response.parts:
                 # Check for safety ratings if available
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                      reason = response.prompt_feedback.block_reason
                      raw_output = f"[Blocked by Google Safety Settings - Prompt: {reason}]"
                 elif hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 'STOP':
                      reason = response.candidates[0].finish_reason
                      ratings = response.candidates[0].safety_ratings
                      raw_output = f"[Blocked by Google Safety Settings - Response: {reason}, Ratings: {ratings}]"
                 else:
                      raw_output = "[Blocked or Empty Response from Google API]"
                 self.logger.warning(f"Google AI response blocked or empty. Prompt: {prompt[:100]}...")
                 output_tokens = 0
            else:
                raw_output = response.text
                # Count output tokens (API doesn't return usage counts directly in generate_content response)
                try: output_tokens = model_to_use.count_tokens(raw_output).total_tokens
                except: output_tokens = len(raw_output.split()) # Fallback estimate

            # Count input tokens
            try: input_tokens = model_to_use.count_tokens(prompt).total_tokens
            except: input_tokens = len(prompt.split()) # Fallback estimate

            memory_used = 0 # N/A

            if generation_time > 0 and output_tokens > 0:
                 tokens_per_sec = output_tokens / generation_time
                 self.logger.info(f"Google AI generated {output_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")
            elif output_tokens > 0:
                 self.logger.info(f"Google AI generated {output_tokens} tokens")
            # Log block reason if available
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 self.logger.warning(f"Google AI Prompt Blocked: {response.prompt_feedback.block_reason}")
            elif hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 'STOP':
                 self.logger.warning(f"Google AI Response Finish Reason: {response.candidates[0].finish_reason}")


        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Google AI API call failed after {generation_time:.2f}s: {e}", exc_info=True)
            raw_output = f"[Google API Error: {e}]"
            try: input_tokens = len(prompt.split())
            except: input_tokens = 0
            output_tokens = 0


        filtered_output = raw_output # Assume no <think> tags

        return GenerationResult(
            raw_output=raw_output,
            filtered_output=filtered_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time=generation_time,
            memory_used=0, # N/A
            device_map={"provider": "google"}
        )

    @log_execution_time()
    def process_with_chat_template(self, system_prompt: str, user_message: str, hyperparams: Optional[HyperparameterConfig] = None, **kwargs) -> GenerationResult:
        """Generate text using Google AI with system prompt."""
        if self.model is None:
            raise RuntimeError("Google AI Model not loaded. Call load_model() first.")

        # Re-initialize the model if the system prompt changes or wasn't set
        # Google's API ties system_instruction to the model object instance
        current_system_instruction = getattr(self.model, 'system_instruction', None)
        # Handle the case where system_instruction might be a Part object
        current_system_text = current_system_instruction.text if hasattr(current_system_instruction, 'text') else current_system_instruction

        if current_system_text != system_prompt:
            self.logger.info(f"Configuring Google AI model with new system prompt: {system_prompt[:50]}...")
            self._system_prompt = system_prompt
            try:
                 # Create a new model instance with the system instruction
                 self.model = genai.GenerativeModel(
                     self.model_specifier,
                     system_instruction=self._system_prompt
                 )
                 # Verify it works
                 self.model.count_tokens("test prompt")
            except Exception as e:
                 self.logger.error(f"Failed to set system prompt for Google AI: {e}. Will prepend to user message.", exc_info=True)
                 # Fallback: create model without system prompt and prepend
                 self.model = genai.GenerativeModel(self.model_specifier)
                 user_message = f"System Instruction: {system_prompt}\n\nUser Request: {user_message}"
                 self._system_prompt = None # Clear internal state if prepending


        # Use the generate method, which will now use the correctly configured self.model
        return self.generate(
            prompt=user_message,
            hyperparams=hyperparams,
            **kwargs
        )

    def unload_model(self):
        # No explicit unloading needed, but clear model reference
        if hasattr(self, 'model') and self.model is not None:
             del self.model
             self.model = None
             self.logger.debug("Google AI model reference cleared.")
        pass

# --- Anthropic (Claude) Backend ---
class AnthropicBackend(LLMBackend):
    """Implementation for Anthropic (Claude) API."""

    def __init__(self, api_key: str, model_specifier: str, hyperparameters: Optional[HyperparameterConfig] = None):
        super().__init__(model_specifier, hyperparameters)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run 'pip install anthropic'")

        self.api_key = api_key
        self.client = None # Initialize in load_model

    @property
    def provider_identifier(self) -> str:
        return "anthropic"

    def load_model(self, **kwargs) -> bool:
        """Initialize the Anthropic client and test the API key."""
        try:
            # Explicitly create an httpx client if needed (e.g., for proxies, timeouts)
            # http_client = httpx.Client(proxies=None, timeout=60.0) # Example: no proxy, 60s timeout
            self.client = anthropic.Anthropic(api_key=self.api_key) # Pass client if created: http_client=http_client
            # Test API key by counting tokens (simple and cheap)
            self.client.count_tokens("test connection")
            self.logger.info(f"Anthropic API client initialized for model {self.model_specifier} and key verified.")
            return True
        except anthropic.AuthenticationError:
            self.logger.error("Anthropic API key is invalid.")
            ConsoleOutput.error("Anthropic API key is invalid. Please check your configuration.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client or verify key: {e}", exc_info=True)
            ConsoleOutput.error(f"Anthropic connection failed: {e}. Check key and network access.")
            return False

    def _map_hyperparams(self, hp: HyperparameterConfig) -> Dict[str, Any]:
        """Maps HyperparameterConfig to Anthropic's API parameters."""
        params = {}

        # Anthropic requires max_tokens
        params["max_tokens"] = hp.max_new_tokens or 4096 # Use a reasonable default if None

        if hp.temperature is not None:
             # Anthropic range [0.0, 1.0]
             params["temperature"] = max(0.0, min(1.0, hp.temperature))
        # Handle do_sample implicitly: Anthropic samples if temperature > 0
        if not hp.do_sample:
             params["temperature"] = 0.0 # Force deterministic

        if hp.top_p is not None:
             # Anthropic range (0.0, 1.0] - exclude 0? Check docs. Assuming similar to OpenAI.
             params["top_p"] = max(0.001, min(1.0, hp.top_p))
        if hp.top_k is not None and hp.top_k > 0:
             params["top_k"] = hp.top_k

        # Map stop sequences if provided
        # if hp.stop_sequences:
        #    params["stop_sequences"] = hp.stop_sequences

        # Note: Anthropic doesn't support repetition_penalty, length_penalty, etc.
        if hasattr(hp, 'repetition_penalty') and hp.repetition_penalty != 1.0:
            self.logger.warning("repetition_penalty is not supported by Anthropic API.")

        return params

    @log_execution_time()
    def generate(self, prompt: str, hyperparams: Optional[HyperparameterConfig] = None, **kwargs) -> GenerationResult:
        """Generate text using Anthropic (treating raw prompt as user message)."""
        if self.client is None:
            raise RuntimeError("Anthropic client not initialized. Call load_model() first.")

        # Use process_with_chat_template as the Messages API is preferred
        # Provide a default system prompt if none is implicitly set
        default_system = "You are a helpful assistant."
        return self.process_with_chat_template(
            system_prompt=default_system, # Use a basic default
            user_message=prompt,
            hyperparams=hyperparams,
            **kwargs
        )

    @log_execution_time()
    def process_with_chat_template(self, system_prompt: str, user_message: str, hyperparams: Optional[HyperparameterConfig] = None, **kwargs) -> GenerationResult:
        """Generate text using Anthropic with system and user prompts (Messages API)."""
        if self.client is None:
            raise RuntimeError("Anthropic client not initialized. Call load_model() first.")

        hp = hyperparams or self.hyperparams
        api_params = self._map_hyperparams(hp)

        messages=[
            {"role": "user", "content": user_message}
        ]

        start_time = time.time()
        raw_output = ""
        input_tokens = 0
        output_tokens = 0
        try:
            response = self.client.messages.create(
                model=self.model_specifier,
                system=system_prompt, # Use the dedicated system parameter
                messages=messages,
                **api_params
            )
            generation_time = time.time() - start_time

            # Handle response structure (content is a list)
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                raw_output = response.content[0].text
            else:
                 raw_output = "[Anthropic: Empty or unexpected response content]"
                 self.logger.warning(f"Anthropic response content was empty or malformed: {response.content}")


            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            memory_used = 0 # N/A
            filtered_output = raw_output

            if generation_time > 0 and output_tokens > 0:
                 tokens_per_sec = output_tokens / generation_time
                 self.logger.info(f"Anthropic generated {output_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")
            elif output_tokens > 0:
                 self.logger.info(f"Anthropic generated {output_tokens} tokens")
            else:
                 self.logger.warning(f"Anthropic generated 0 tokens. Finish Reason: {response.stop_reason}")

        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Anthropic API call failed after {generation_time:.2f}s: {e}", exc_info=True)
            raw_output = f"[Anthropic Error: {e}]"
            try: input_tokens = len(system_prompt.split()) + len(user_message.split())
            except: input_tokens = 0
            output_tokens = 0

        return GenerationResult(
            raw_output=raw_output,
            filtered_output=filtered_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time=generation_time,
            memory_used=0, # N/A
            device_map={"provider": "anthropic"}
        )

    def unload_model(self):
        # No explicit unloading needed, clear client reference
        if hasattr(self, 'client') and self.client is not None:
             del self.client
             self.client = None
             self.logger.debug("Anthropic client reference cleared.")
        pass


# --- Backend Factory ---

def get_llm_backend(provider: str,
                      model_specifier: str,
                      api_keys: Dict[str, str],
                      hyperparameters: HyperparameterConfig,
                      # Kwargs for local-specific configs
                      # Add model_config required by LocalTransformerBackend
                      model_config: Optional[ModelConfig] = None,
                      quantization_config: Optional[QuantizationConfig] = None,
                      layer_split_config: Optional[LayerSplitConfig] = None,
                      memory_config: Optional[MemoryConfig] = None # Keep for legacy compatibility
                      ) -> Optional[LLMBackend]:
    """Factory function to create the appropriate LLM backend."""
    provider_lower = provider.lower()
    logger.info(f"Attempting to create backend for provider: {provider_lower}, model: {model_specifier}")

    try:
        if provider_lower == "local":
            # This is a local Hugging Face Transformers model
            # Ensure model_config is provided
            if model_config is None:
                 logger.warning(f"ModelConfig not provided for local model {model_specifier}. Attempting to fetch from registry.")
                 model_entry = get_model_config(model_specifier)
                 if model_entry:
                      # Convert ModelEntry to ModelConfig
                      model_config = ModelConfig(
                          name=model_entry.name, model_id=model_entry.model_id,
                          supports_thinking=model_entry.supports_thinking,
                          thinking_tokens=model_entry.thinking_tokens or [],
                          max_context=model_entry.max_context,
                          optimal_chunk_size=model_entry.optimal_chunk_size,
                          temperature=model_entry.temperature, top_p=model_entry.top_p,
                          max_new_tokens=model_entry.max_new_tokens,
                          quantization_support=model_entry.quantization_support or ["4bit", "8bit"]
                      )
                 else:
                      logger.error(f"Model {model_specifier} not found in registry and no ModelConfig provided.")
                      raise ValueError(f"Missing ModelConfig for local model {model_specifier}")


            return LocalTransformerBackend(
                model_id=model_specifier,
                model_config=model_config, # Pass the resolved ModelConfig
                quantization_config=quantization_config,
                layer_split_config=layer_split_config,
                hyperparameters=hyperparameters,
                memory_config=memory_config # Pass legacy if needed
            )

        elif provider_lower == "local_gguf":
             if not LLAMACPP_AVAILABLE:
                 raise ImportError("llama-cpp-python not found. Cannot use GGUF backend.")

             # model_specifier is the path for GGUF
             gguf_model_path = Path(model_specifier)
             if not gguf_model_path.is_file():
                 raise FileNotFoundError(f"GGUF model file not found: {gguf_model_path}")

             # Create LlamaCppConfig - use LayerSplitConfig if available for n_gpu_layers
             gpu_layers = layer_split_config.gpu_layers if layer_split_config else -1

             gguf_config = LlamaCppConfig(
                 model_path=gguf_model_path,
                 n_ctx=hyperparameters.max_length or 2048, # Use max_length from hyperparams
                 n_gpu_layers=gpu_layers,
                 n_batch=512, # Default, make configurable?
                 # Add other relevant LlamaCpp settings if needed
             )
             # Directly return the LlamaCppBackend instance (from llamacpp_backend.py)
             return LlamaCppBackend(
                 config=gguf_config,
                 hyperparameters=hyperparameters
             )


        elif provider_lower == "openai":
            if not OPENAI_AVAILABLE:
                 raise ImportError("OpenAI library not found. Please run 'pip install openai'")
            api_key = api_keys.get("openai")
            if not api_key:
                raise ValueError("OpenAI API key not configured.")
            return OpenAIBackend(
                api_key=api_key,
                model_specifier=model_specifier,
                hyperparameters=hyperparameters
            )

        elif provider_lower == "google":
            if not GOOGLE_AI_AVAILABLE:
                 raise ImportError("Google Generative AI library not found. Please run 'pip install google-generativeai'")
            api_key = api_keys.get("google")
            if not api_key:
                raise ValueError("Google API key not configured.")
            return GoogleAIBackend(
                api_key=api_key,
                model_specifier=model_specifier,
                hyperparameters=hyperparameters
            )

        elif provider_lower == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                 raise ImportError("Anthropic library not found. Please run 'pip install anthropic'")
            api_key = api_keys.get("anthropic")
            if not api_key:
                raise ValueError("Anthropic API key not configured.")
            return AnthropicBackend(
                api_key=api_key,
                model_specifier=model_specifier,
                hyperparameters=hyperparameters
            )

        # --- Add other cloud providers here ---
        # elif provider_lower == "cohere":
        #     # ... implementation ...
        # elif provider_lower == "openrouter":
        #     # ... implementation ...

        else:
            logger.error(f"Unsupported provider: '{provider_lower}'.")
            raise ValueError(f"Unsupported provider: {provider}")

    except (ImportError, ValueError, TypeError, FileNotFoundError) as e:
        logger.error(f"Failed to create backend for {provider}: {e}", exc_info=False) # Reduce noise for common errors
        ConsoleOutput.error(f"Error initializing backend '{provider}': {e}")
        return None # Return None on failure


# --- BatchProcessor (Now uses LLMBackend) ---
class BatchProcessor:
    """Process text in batches for efficiency"""

    def __init__(self, llm_backend: LLMBackend):
        self.llm_backend = llm_backend
        self.logger = get_logger_conf(f"{__name__}.BatchProcessor")

    def process_batch(self,
                      texts: List[str],
                      system_prompt: str,
                      batch_size: int = 1, # Batching is complex with heterogeneous backends, default to 1
                      hyperparams: Optional[HyperparameterConfig] = None,
                      **generation_kwargs) -> List[GenerationResult]:
        """
        Process multiple texts in batches (currently serial)

        Args:
            texts: List of texts to process
            system_prompt: System prompt for all texts
            batch_size: (Note: True batching only supported by LocalTransformerBackend)
            **generation_kwargs: Additional generation parameters

        Returns:
            List of GenerationResults
        """
        # TODO: Implement true batching for LocalTransformerBackend if needed
        # For API backends, batching often means parallel requests, not a single batch input

        results = []

        self.logger.info(f"Processing {len(texts)} texts sequentially (batch_size={batch_size} ignored for non-local-transformer backends)")

        # Ensure text backend is loaded before starting the loop
        if not self.llm_backend.model: # Check if model is loaded (relevant for local)
             self.logger.info("Loading text backend model before batch processing...")
             if not self.llm_backend.load_model():
                  self.logger.error("Failed to load model for batch processing. Aborting.")
                  # Return error results for all items
                  return [GenerationResult(raw_output="", filtered_output=f"Error: Backend failed to load",
                                           input_tokens=0, output_tokens=0, generation_time=0, memory_used=0,
                                           device_map={"provider": "error"}) for _ in texts]


        with LoggingProgress(self.logger, "Processing batch", len(texts)) as progress:
            for i, text in enumerate(texts):
                self.logger.debug(f"Processing item {i+1}/{len(texts)}")

                try:
                    result = self.llm_backend.process_with_chat_template(
                        system_prompt,
                        text,
                        hyperparams=hyperparams,
                        **generation_kwargs
                    )
                    results.append(result)

                except Exception as e:
                    self.logger.error(f"Failed to process text item {i}: {e}", exc_info=True)
                    # Create error result
                    results.append(GenerationResult(
                        raw_output="",
                        filtered_output=f"Error: {str(e)}",
                        input_tokens=0,
                        output_tokens=0,
                        generation_time=0,
                        memory_used=0,
                        device_map={"provider": "error"}
                    ))

                # Optional: Clear cache between items (if local GPU and memory is tight)
                # if self.llm_backend.provider_identifier.startswith("local") and torch.cuda.is_available():
                #    torch.cuda.empty_cache()

                progress.update(1, f"Item {i+1}/{len(texts)} completed")
                # ConsoleOutput.progress_bar(i+1, len(texts), prefix="Processing") # Can be noisy with LoggingProgress

        self.logger.info(f"Batch processing complete: {len(results)} results")
        return results
