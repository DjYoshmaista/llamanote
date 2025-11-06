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
        AutoConfig,
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
from ...utils.memory_manager import (
    CUDAMemoryManager,
    OOMRecoveryStrategy,
    with_oom_handling,
    log_memory_summary
)
from ...config.manager import ConfigManager # Import ConfigManager
from ...config.settings import DEFAULT_CACHE_DIR, DEFAULT_OFFLOAD_DIR
from ..hybrid_kv_cache import HybridKVCache, CacheConfig
from ..attention_patcher import AttentionPatcher
from ..sliding_window_attention import SlidingWindowConfig
from ..cache import LlamaNoteDynamicCache, CacheStrategyConfig, MemoryStrategy
from ..cache.generation_wrapper import CachedGenerationWrapper

logger = get_logger_conf(__name__)

# --- Model Loader Helper (Refactoring Item 5) ---

class LocalModelLoader:
    """Helper class to consolidate the logic for loading HF Transformers models."""
    
    def __init__(self, model_id: str, quant_config: QuantizationConfig, 
                 split_config: LayerSplitConfig, trust_remote_code: bool, config_manager: ConfigManager):
        self.model_id = model_id
        self.quant_config = quant_config
        self.split_config = split_config
        self.trust_remote_code = trust_remote_code
        self.config_manager = config_manager
        self.model_cache_dir = self.config_manager.get_dir("model_cache")
        self.device_manager = get_device_manager()
        self.load_config = {}

    def build_load_config(self) -> Dict[str, Any]:
        """Builds the kwargs dictionary for AutoModelForCausalLM.from_pretrained."""
        self.load_config = {
            "cache_dir": str(self.model_cache_dir),
            "trust_remote_code": self.trust_remote_code
        }
        
        is_cpu = not self.device_manager.is_cuda_available()

        # 1. Set Device Map & Offloading
        if is_cpu:
            self.load_config["device_map"] = {"": "cpu"}
            self.load_config["low_cpu_mem_usage"] = False
            logger.info("Configuring model for CPU-only load.")
        elif self.split_config.enabled:
            # Try to load saved layer split configuration
            from ...utils.device_map_builder import LayerSplitConfigManager
            from ...utils.auto_layer_split import auto_discover_on_load

            split_mgr = LayerSplitConfigManager()

            # Extract GPU memory from config
            gpu_memory_gb = 4.0  # Default
            if self.split_config.max_gpu_memory:
                gpu_mem_str = list(self.split_config.max_gpu_memory.values())[0]
                # Handle both "GB" and "GiB" suffixes (case-insensitive)
                gpu_mem_str_upper = gpu_mem_str.upper()
                if "GIB" in gpu_mem_str_upper:
                    gpu_memory_gb = float(gpu_mem_str_upper.replace("GIB", "").strip())
                elif "GB" in gpu_mem_str_upper:
                    gpu_memory_gb = float(gpu_mem_str_upper.replace("GB", "").strip())
                else:
                    # Try to parse as plain number
                    gpu_memory_gb = float(gpu_mem_str.strip())

            # Try to load saved split configuration
            saved_splits = split_mgr.load_config(
                model_id=self.model_id,
                quant_method=self.quant_config.method,
                gpu_memory_gb=gpu_memory_gb
            )

            # If no saved split, trigger automatic discovery
            if not saved_splits and self.split_config.auto_discover_splits:
                logger.info("No saved layer split found - triggering automatic discovery")

                # Get quantization config for discovery
                quant_config_for_discovery = self.quant_config.to_bnb_config() if self.quant_config.method != "none" else None

                try:
                    saved_splits = auto_discover_on_load(
                        model_id=self.model_id,
                        quant_method=self.quant_config.method,
                        gpu_memory_gb=gpu_memory_gb,
                        quantization_config=quant_config_for_discovery,
                        force_rediscover=False
                    )
                except Exception as e:
                    logger.warning(f"Automatic discovery failed: {e}")
                    saved_splits = None

            if saved_splits:
                # Use saved explicit device map (minimum split = most GPU layers)
                min_split, max_split = saved_splits
                self.load_config["device_map"] = min_split.device_map
                self.load_config["low_cpu_mem_usage"] = True
                logger.info(f"Using saved layer split: {min_split.layers_on_gpu} GPU / {min_split.layers_on_cpu} CPU layers")
                logger.info(f"Explicit device map loaded from cache (estimated GPU memory: {min_split.gpu_memory_used_mb:.1f}MB)")
            else:
                # Fall back to auto with max_memory constraint
                self.load_config["device_map"] = "auto"
                self.load_config["max_memory"] = self.split_config.get_max_memory_dict()
                self.load_config["low_cpu_mem_usage"] = True
                logger.info(f"No saved split found, using device_map='auto' with max_memory: {self.load_config['max_memory']}")
                logger.info("Tip: Run LayerSplitFinder to discover and save optimal splits for this model")

            # Add offload folder if configured
            offload_dir = self.split_config.offload_folder or DEFAULT_OFFLOAD_DIR
            if self.split_config.offload_state_dict or offload_dir:
                 self.load_config["offload_folder"] = str(offload_dir)
                 self.load_config["offload_state_dict"] = self.split_config.offload_state_dict
        else:
            self.load_config["device_map"] = "auto"
            self.load_config["low_cpu_mem_usage"] = True
            logger.info("Configuring model for 'auto' device mapping (no explicit splitting).")

        # 2. Set Quantization
        bnb_config = self.quant_config.to_bnb_config() if not is_cpu else None
        if bnb_config:
            self.load_config["quantization_config"] = bnb_config
            self.load_config["dtype"] = self.quant_config.compute_dtype
            logger.info(f"Applying {self.quant_config.method} quantization with {self.quant_config.compute_dtype}.")
        elif not is_cpu:
            self.load_config["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info(f"Using default GPU dtype: {self.load_config['dtype']}")
        else:
             self.load_config["dtype"] = torch.float32
             logger.info("Using default CPU dtype: float32")

        # 3. Handle potential 16bit method (which just means setting dtype)
        if self.quant_config.method == "16bit" and "quantization_config" not in self.load_config:
            self.load_config["dtype"] = self.quant_config.compute_dtype or torch.float16
            logger.info(f"Using 16-bit precision: {self.load_config['dtype']}")

        return self.load_config

    def _get_model_num_layers(self) -> Optional[int]:
        """
        Get the number of layers in the model from config.

        Returns:
            Number of layers, or None if unable to determine
        """
        try:
            config = AutoConfig.from_pretrained(
                self.model_id,
                cache_dir=str(self.model_cache_dir),
                trust_remote_code=self.trust_remote_code
            )

            # Try common attribute names for layer count
            for attr in ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers']:
                if hasattr(config, attr):
                    num_layers = getattr(config, attr)
                    logger.info(f"Detected {num_layers} layers in model {self.model_id}")
                    return num_layers

            logger.warning(f"Could not determine number of layers for {self.model_id}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get model config for layer count: {e}")
            return None

    def load(self) -> Tuple[Any, Any]:
        """Attempts to load the model and tokenizer."""
        if not TORCH_AVAILABLE:
            raise ModelLoadError("PyTorch/Transformers not installed.", self.model_id)

        # This is a workaround for a bug in some versions of `transformers` that
        # causes a crash when loading tokenizers for models without chat template files.
        # We temporarily disable the function that looks for remote template files.
        original_list_repo_templates = None
        try:
            from transformers.utils import hub
            if hasattr(hub, 'list_repo_templates'):
                original_list_repo_templates = hub.list_repo_templates
                hub.list_repo_templates = lambda *args, **kwargs: []
                logger.info("Temporarily patched `transformers.utils.hub.list_repo_templates` to avoid chat template errors.")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not patch `list_repo_templates`: {e}. Proceeding without patch.")

        try:
            load_config = self.build_load_config()

            # --- Display Memory Projection (if enabled) ---
            if self.split_config.show_memory_projection:
                from ...utils.memory_estimator import display_memory_projection
                logger.info(f"Displaying memory projection for {self.model_id}")
                try:
                    projection = display_memory_projection(
                        model_id=self.model_id,
                        quantization=self.quant_config.method,
                        max_seq_length=4096,  # TODO: Get from hyperparameters
                        cache_dir=self.model_cache_dir,
                        trust_remote_code=self.trust_remote_code
                    )
                    logger.info(f"Memory projection: Model={projection.model_size_mb:.0f}MB, "
                               f"KV-Cache={projection.kv_cache_size_mb:.0f}MB, "
                               f"Total={projection.total_size_mb:.0f}MB")
                except Exception as e:
                    logger.warning(f"Could not display memory projection: {e}")

            # --- Load Tokenizer ---
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    cache_dir=str(self.model_cache_dir),
                    use_fast=True,
                    trust_remote_code=self.trust_remote_code
                )
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {self.model_id}: {e}", exc_info=True)
                raise ModelLoadError(f"Failed to load tokenizer: {e}", self.model_id) from e

            # --- Load Model with OOM Handling ---
            if self.device_manager.is_cuda_available():
                log_memory_summary("before model loading", logger)

            def load_model_fn(**strategy_params):
                config = load_config.copy()
                if 'max_memory' in strategy_params:
                    config['max_memory'] = strategy_params['max_memory']
                if 'device_map' in strategy_params:
                    config['device_map'] = strategy_params['device_map']
                if 'low_cpu_mem_usage' in strategy_params:
                    config['low_cpu_mem_usage'] = strategy_params['low_cpu_mem_usage']
                if 'offload_state_dict' in strategy_params:
                    config['offload_state_dict'] = strategy_params['offload_state_dict']
                logger.info(f"Loading model with config: device_map={config.get('device_map')}, "
                           f"max_memory={config.get('max_memory')}")
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **config)
                return model

            try:
                if self.split_config.auto_oom_handling and self.device_manager.is_cuda_available():
                    logger.info("🛡️  Automatic OOM handling enabled for LLM")
                    model_num_layers = self._get_model_num_layers()
                    initial_gpu_mem = list(self.split_config.max_gpu_memory.values())[0] if self.split_config.max_gpu_memory else "4GB"
                    recovery_strategy = OOMRecoveryStrategy(
                        initial_gpu_memory=initial_gpu_mem,
                        initial_cpu_memory=self.split_config.max_cpu_memory,
                        min_gpu_memory="1GB",
                        model_num_layers=model_num_layers,
                        use_iterative_layer_split=True,
                        offload_folder=self.split_config.offload_folder,
                        model_id=self.model_id,
                        use_explicit_device_map=True,
                        logger=logger
                    )
                    try:
                        model, final_params = with_oom_handling(
                            load_fn=load_model_fn,
                            recovery_strategy=recovery_strategy,
                            logger=logger
                        )
                        if final_params:
                            logger.info(f"✓ Model loaded with adjusted parameters: {final_params}")
                    except Exception as e:
                        if CUDAMemoryManager.is_oom_error(e):
                            logger.error("Failed to load model: CUDA Out of Memory after all recovery attempts")
                            logger.error("Consider: 1) Using smaller model, 2) Enabling disk offloading, 3) Using GGUF format")
                        raise
                else:
                    logger.info(f"Attempting to load model '{self.model_id}'...")
                    model = load_model_fn()

                logger.info(f"✓ Successfully loaded model: {self.model_id}")
                if self.device_manager.is_cuda_available():
                    log_memory_summary("after model loading", logger)
                return model, tokenizer

            except Exception as e:
                logger.error(f"Failed to load model {self.model_id}: {e}", exc_info=True)
                is_oom = CUDAMemoryManager.is_oom_error(e)
                if is_oom:
                    logger.warning("OOM detected - skipping GPU fallback attempts, trying CPU-only load.")
                    if load_config.get("device_map") != {"": "cpu"}:
                         cpu_config = {
                            "cache_dir": str(self.model_cache_dir),
                            "trust_remote_code": self.trust_remote_code,
                            "dtype": torch.float32,
                            "device_map": {"": "cpu"},
                         }
                         try:
                              model = AutoModelForCausalLM.from_pretrained(self.model_id, **cpu_config)
                              logger.info("Successfully loaded model with CPU-only fallback.")
                              return model, tokenizer
                         except Exception as e3:
                              logger.error(f"CPU-only fallback failed: {e3}", exc_info=True)
                else:
                    if load_config.get("device_map") != {"": "cpu"} and (load_config.get("quantization_config") or load_config.get("max_memory")):
                        logger.warning("Falling back to standard 'auto' device map without quantization/limits.")
                        fallback_config = {
                            "cache_dir": str(self.model_cache_dir),
                            "trust_remote_code": self.trust_remote_code,
                            "dtype": torch.bfloat16 if self.device_manager.is_cuda_available() else torch.float32,
                            "device_map": "auto",
                            "low_cpu_mem_usage": self.device_manager.is_cuda_available(),
                        }
                        try:
                            model = AutoModelForCausalLM.from_pretrained(self.model_id, **fallback_config)
                            logger.info("Successfully loaded model with standard 'auto' fallback.")
                            return model, tokenizer
                        except Exception as e2:
                            logger.error(f"Standard 'auto' fallback failed: {e2}", exc_info=True)
                if load_config.get("device_map") != {"": "cpu"}:
                     logger.warning("Falling back to CPU-only load.")
                     cpu_config = {
                        "cache_dir": str(self.model_cache_dir),
                        "trust_remote_code": self.trust_remote_code,
                        "dtype": torch.float32,
                        "device_map": {"": "cpu"},
                     }
                     try:
                          model = AutoModelForCausalLM.from_pretrained(self.model_id, **cpu_config)
                          logger.info("Successfully loaded model with CPU-only fallback.")
                          return model, tokenizer
                     except Exception as e3:
                          logger.error(f"CPU-only fallback failed: {e3}", exc_info=True)
                raise ModelLoadError(f"All loading attempts failed. Last error: {e}", self.model_id) from e

        finally:
            # --- Restore monkey-patch ---
            if original_list_repo_templates:
                from transformers.utils import hub
                hub.list_repo_templates = original_list_repo_templates
                logger.info("Restored original `transformers.utils.hub.list_repo_templates`.")

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

        # Initialize advanced cache system (NEW INTEGRATED SYSTEM)
        self.generation_wrapper: Optional[CachedGenerationWrapper] = None

        if self.split_config.use_advanced_cache:
            self.logger.info(f"Advanced cache system enabled: strategy={self.split_config.cache_strategy}")

        # Legacy cache systems (kept for backward compatibility)
        self.hybrid_cache: Optional[HybridKVCache] = None
        if self.split_config.use_hybrid_kv_cache and not self.split_config.use_advanced_cache:
            cache_config = CacheConfig(
                hot_cache_max_size_mb=self.split_config.kv_cache_hot_size_mb,
                cold_cache_max_size_mb=self.split_config.kv_cache_cold_size_mb,
                device_hot="cuda" if self.device_manager.is_cuda_available() else "cpu",
                device_cold="cpu"
            )
            self.hybrid_cache = HybridKVCache(config=cache_config)
            self.logger.info(f"Legacy hybrid KV-Cache enabled (hot={cache_config.hot_cache_max_size_mb}MB)")

        self.attention_patcher: Optional[AttentionPatcher] = None
        if self.split_config.use_sliding_window and not self.split_config.use_advanced_cache:
            sliding_config = SlidingWindowConfig(
                window_size=self.split_config.sliding_window_size,
                stride=self.split_config.sliding_window_stride,
                keep_prefix_tokens=self.split_config.sliding_window_keep_prefix,
                device="cuda" if self.device_manager.is_cuda_available() else "cpu"
            )
            self.attention_patcher = AttentionPatcher(sliding_window_config=sliding_config)
            self.logger.info(f"Legacy sliding window configured (window={sliding_config.window_size})")

        self.logger.info(f"Initialized LocalHFBackend for {self.model_id}")

    @log_execution_time(logger_name=__name__)
    @log_resource_usage(logger_name=__name__)
    def load(self, trust_remote_code: bool = True, **kwargs) -> bool:
        """Loads the model and tokenizer using the ModelLoader helper."""
        if self.is_loaded and self.model_handle is not None:
            self.logger.info("Model is already loaded.")
            return True
        
        config_manager = ConfigManager()
        loader = LocalModelLoader(
            self.model_specifier,
            self.quant_config,
            self.split_config,
            trust_remote_code,
            config_manager
        )        
        try:
            model, tokenizer = loader.load()
            self.model_handle = model
            self.tokenizer = tokenizer
            self._post_load_setup()
            self._log_model_info()

            # Initialize generation wrapper with advanced cache if enabled
            if self.split_config.use_advanced_cache:
                self.generation_wrapper = CachedGenerationWrapper(
                    model=self.model_handle,
                    strategy=self.split_config.cache_strategy,
                    enable_hybrid_cache=True, # You can make this configurable
                    enable_sliding_window=True # You can make this configurable
                )
                self.logger.info("Generation wrapper initialized with advanced cache")

            # Apply legacy sliding window patch if enabled (backward compatibility)
            if self.attention_patcher:
                self.logger.info("Applying sliding window attention patch...")
                if self.attention_patcher.patch_model(self.model_handle, model_type="auto"):
                    self.logger.info("Successfully patched model with sliding window attention")
                else:
                    self.logger.warning("Failed to patch model with sliding window attention")

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

        # Generate - use wrapper if available for advanced cache integration
        start_time = time.time()
        with torch.no_grad():
            if self.generation_wrapper:
                # Use advanced cache-integrated generation
                outputs = self.generation_wrapper.generate(
                    **inputs,
                    generation_config=gen_config
                )
            else:
                # Standard generation
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

    def get_cache_statistics(self) -> Optional[Dict[str, Any]]:
        """Get hybrid cache statistics if cache is enabled."""
        if self.hybrid_cache:
            return self.hybrid_cache.get_statistics()
        return None

    def log_cache_statistics(self):
        """Log hybrid cache statistics if cache is enabled."""
        if self.hybrid_cache:
            self.logger.info(self.hybrid_cache.format_statistics())

    def log_attention_statistics(self):
        """Log sliding window attention statistics if enabled."""
        if self.attention_patcher:
            self.logger.info(self.attention_patcher.format_statistics())

    def log_advanced_cache_statistics(self):
        """Log advanced cache statistics if enabled."""
        if self.generation_wrapper:
            self.generation_wrapper.log_cache_statistics()

    def unload(self):
        """Unload model and tokenizer, clear memory."""
        if not self.is_loaded:
            return
        self.logger.info(f"Unloading local HF model: {self.model_specifier}")

        # Log advanced cache statistics if enabled (NEW SYSTEM)
        if hasattr(self, 'generation_wrapper') and self.generation_wrapper:
            self.log_advanced_cache_statistics()
            if hasattr(self.generation_wrapper, 'cache') and hasattr(self.generation_wrapper.cache, 'reset'):
                self.generation_wrapper.cache.reset()

        # Log legacy cache statistics before unloading (BACKWARD COMPATIBILITY)
        if hasattr(self, 'hybrid_cache') and self.hybrid_cache:
            self.log_cache_statistics()
            self.hybrid_cache.clear()

        # Log attention statistics and unpatch if enabled (LEGACY)
        if hasattr(self, 'attention_patcher') and self.attention_patcher:
            self.log_attention_statistics()
            self.attention_patcher.unpatch_model()
            self.attention_patcher.reset_statistics()

        cleanup_resources([self.model_handle, self.tokenizer], clear_cuda=True)
        self.model_handle = None
        self.tokenizer = None
        self.device_map = None
        if hasattr(self, 'generation_wrapper'):
            self.generation_wrapper = None
        self.is_loaded = False
        self.logger.info("Local HF model unloaded.")
