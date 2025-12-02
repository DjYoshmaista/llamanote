# llamanote/utils/layer_split_finder.py
"""
Layer Split Finder - Automatically discovers optimal layer splits

Tests different layer split configurations to find:
1. Minimum split (most layers on GPU) that works
2. Maximum split (fewest layers on GPU) that still uses GPU

Saves successful configurations for reuse.
"""

import torch
import gc
from pathlib import Path
from typing import Optional, Tuple, Callable
from transformers import AutoModelForCausalLM

from .device_map_builder import DeviceMapBuilder, LayerSplitResult, LayerSplitConfigManager
from .memory_manager import CUDAMemoryManager
from .logger import get_logger_conf

logger = get_logger_conf(__name__)


class LayerSplitFinder:
    """
    Finds optimal layer splits through binary search and testing.

    This automates the process of finding which layer splits work
    for a given model, quantization, and GPU memory configuration.
    """

    def __init__(self,
                 model_id: str,
                 cache_dir: Optional[Path] = None,
                 trust_remote_code: bool = True):
        """
        Initialize the layer split finder.

        Args:
            model_id: Hugging Face model ID
            cache_dir: Cache directory
            trust_remote_code: Whether to trust remote code
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.logger = logger

        # Initialize device map builder
        self.builder = DeviceMapBuilder(model_id, cache_dir, trust_remote_code)
        self.num_layers = self.builder.num_layers

        # Initialize config manager
        self.config_manager = LayerSplitConfigManager()

    def test_split(self,
                  layers_on_gpu: int,
                  load_config: dict,
                  test_forward_pass: bool = True) -> Tuple[bool, Optional[float]]:
        """
        Test if a specific layer split works.

        Args:
            layers_on_gpu: Number of layers to put on GPU
            load_config: Base configuration for model loading
            test_forward_pass: Whether to test a forward pass

        Returns:
            Tuple of (success, gpu_memory_used_mb)
        """
        if not torch.cuda.is_available():
            self.logger.warning("No CUDA available, skipping test")
            return False, None

        try:
            # Clear GPU memory
            CUDAMemoryManager.clear_cache(aggressive=True)
            gc.collect()

            # Build device map
            device_map = self.builder.build_device_map(layers_on_gpu)

            # Update load config with device map
            test_config = load_config.copy()
            test_config['device_map'] = device_map

            # Enable CPU offload for quantized models when layers are on CPU
            if 'quantization_config' in test_config and layers_on_gpu < self.num_layers:
                # BitsAndBytes requires this flag when offloading quantized layers to CPU
                quant_config = test_config['quantization_config']
                if hasattr(quant_config, 'llm_int8_enable_fp32_cpu_offload'):
                    quant_config.llm_int8_enable_fp32_cpu_offload = True
                elif hasattr(quant_config, 'bnb_4bit_compute_dtype'):
                    # For 4-bit, we need to recreate the config with the flag
                    from transformers import BitsAndBytesConfig
                    test_config['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=getattr(quant_config, 'load_in_4bit', False),
                        load_in_8bit=getattr(quant_config, 'load_in_8bit', False),
                        llm_int8_enable_fp32_cpu_offload=True,
                        bnb_4bit_compute_dtype=getattr(quant_config, 'bnb_4bit_compute_dtype', 'float16'),
                        bnb_4bit_use_double_quant=getattr(quant_config, 'bnb_4bit_use_double_quant', True),
                        bnb_4bit_quant_type=getattr(quant_config, 'bnb_4bit_quant_type', 'nf4')
                    )

            self.logger.info(f"Testing split: {layers_on_gpu}/{self.num_layers} layers on GPU")

            # Try to load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **test_config
            )

            # Get GPU memory usage
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)
            self.logger.info(f"✓ Model loaded: {gpu_memory_mb:.1f}MB GPU memory used")

            # Optionally test forward pass
            if test_forward_pass:
                try:
                    # Create dummy input on correct device
                    first_device = next(model.parameters()).device
                    dummy_input = torch.randint(0, 1000, (1, 10), device=first_device)

                    with torch.no_grad():
                        _ = model(dummy_input)

                    self.logger.info("✓ Forward pass successful")
                except Exception as e:
                    self.logger.warning(f"Forward pass failed: {e}")
                    # Don't fail the test for forward pass issues
                    pass

            # Clean up
            del model
            CUDAMemoryManager.clear_cache(aggressive=True)
            gc.collect()

            return True, gpu_memory_mb

        except Exception as e:
            # Check if OOM error
            is_oom = CUDAMemoryManager.is_oom_error(e)
            if is_oom:
                self.logger.info(f"✗ OOM with {layers_on_gpu} GPU layers")
            else:
                self.logger.error(f"✗ Error testing split: {e}")

            # Clean up
            CUDAMemoryManager.clear_cache(aggressive=True)
            gc.collect()

            return False, None

    def _validate_split_for_runtime(
        self,
        layers_on_gpu: int,
        batch_size: int,
        max_tokens: int,
        model_config: dict,
        quantization: str,
        safety_margin: float = 1.2
    ) -> Tuple[bool, float]:
        """
        Validate layer split against runtime parameters using memory estimation.

        Args:
            layers_on_gpu: Number of layers to put on GPU
            batch_size: Batch size for validation
            max_tokens: Max tokens for validation
            model_config: Model configuration (hidden_size, num_heads, etc.)
            quantization: Quantization method ("4bit", "8bit", "16bit", "none")
            safety_margin: Safety margin multiplier (default: 1.2 = 20% buffer)

        Returns:
            Tuple of (is_valid, estimated_peak_memory_mb)
        """
        from .memory_estimator import MemoryEstimator

        # 1. Estimate model size on GPU
        model_size_mb = MemoryEstimator.estimate_model_size(
            model_id=self.model_id,
            quantization=quantization,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code
        )

        # Scale by fraction of layers on GPU
        gpu_model_size = model_size_mb * (layers_on_gpu / self.num_layers) if self.num_layers > 0 else model_size_mb

        # 2. Estimate KV cache size WITH ACTUAL BATCH_SIZE AND MAX_TOKENS
        kv_cache_mb = MemoryEstimator.estimate_kv_cache_size(
            num_layers=layers_on_gpu,  # Only GPU layers contribute to VRAM KV cache
            hidden_size=model_config.get("hidden_size", 4096),
            num_attention_heads=model_config.get("num_attention_heads", 32),
            max_seq_length=max_tokens + 512,  # max_tokens + average prompt length
            batch_size=batch_size,  # ACTUAL BATCH SIZE
            dtype_bytes=2  # fp16/bfloat16
        )

        # 3. Estimate context processing memory
        context_mb = MemoryEstimator.estimate_context_size(
            max_seq_length=max_tokens + 512,
            hidden_size=model_config.get("hidden_size", 4096),
            batch_size=batch_size,
            dtype_bytes=2
        )

        # 4. Calculate total estimated memory
        estimated_peak = gpu_model_size + kv_cache_mb + context_mb

        # 5. Add safety margin for overhead and variations
        estimated_peak_with_margin = estimated_peak * safety_margin

        # 6. Check against available VRAM
        if not torch.cuda.is_available():
            self.logger.warning("No CUDA available, validation skipped")
            return True, estimated_peak_with_margin

        available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        usable_vram = available_vram * 0.85  # Use up to 85% to avoid instability

        is_valid = estimated_peak_with_margin < usable_vram

        self.logger.debug(
            f"Memory validation for {layers_on_gpu} GPU layers: "
            f"Model={gpu_model_size:.0f}MB, KV={kv_cache_mb:.0f}MB, Context={context_mb:.0f}MB, "
            f"Total={estimated_peak_with_margin:.0f}MB, Available={usable_vram:.0f}MB, Valid={is_valid}"
        )

        return is_valid, estimated_peak_with_margin

    def _find_valid_split_for_runtime(
        self,
        initial_layers: int,
        batch_size: int,
        max_tokens: int,
        model_config: dict,
        quantization: str,
        safety_margin: float = 1.2
    ) -> int:
        """
        Find valid layer split that passes runtime validation.

        Reduces layers from initial_layers until validation passes.

        Args:
            initial_layers: Starting number of GPU layers
            batch_size: Batch size for validation
            max_tokens: Max tokens for validation
            model_config: Model configuration dict
            quantization: Quantization method
            safety_margin: Safety margin multiplier

        Returns:
            Number of GPU layers that pass validation
        """
        layers = initial_layers

        while layers > 0:
            is_valid, estimated_mem = self._validate_split_for_runtime(
                layers_on_gpu=layers,
                batch_size=batch_size,
                max_tokens=max_tokens,
                model_config=model_config,
                quantization=quantization,
                safety_margin=safety_margin
            )

            if is_valid:
                self.logger.info(f"✓ Runtime validation passed for {layers} GPU layers (~{estimated_mem:.0f}MB)")
                return layers

            self.logger.debug(f"✗ Runtime validation failed for {layers} GPU layers, trying {layers-2}...")
            layers -= 2  # Reduce by 2 layers at a time

        self.logger.warning("No GPU layer configuration passed runtime validation")
        return 0

    def _find_conservative_split(
        self,
        max_layers: int,
        target_vram_usage: float,
        batch_size: int,
        max_tokens: int,
        model_config: dict,
        quantization: str
    ) -> int:
        """
        Find conservative layer split targeting specific VRAM usage percentage.

        Args:
            max_layers: Maximum GPU layers known to work
            target_vram_usage: Target VRAM usage (0.0-1.0, e.g., 0.5 = 50%)
            batch_size: Batch size for validation
            max_tokens: Max tokens for validation
            model_config: Model configuration dict
            quantization: Quantization method

        Returns:
            Number of GPU layers for conservative configuration
        """
        if not torch.cuda.is_available():
            return max_layers  # Can't be conservative without GPU info

        available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        target_memory = available_vram * target_vram_usage

        # Binary search for layers that fit target memory
        left, right = 1, max_layers
        best_layers = 1

        while left <= right:
            mid = (left + right) // 2

            is_valid, estimated_mem = self._validate_split_for_runtime(
                layers_on_gpu=mid,
                batch_size=batch_size,
                max_tokens=max_tokens,
                model_config=model_config,
                quantization=quantization,
                safety_margin=1.2
            )

            if estimated_mem <= target_memory:
                # This fits in target, try more layers
                best_layers = mid
                left = mid + 1
            else:
                # Too much memory, try fewer layers
                right = mid - 1

        self.logger.info(f"Conservative split: {best_layers} GPU layers (~{target_vram_usage*100:.0f}% VRAM target)")
        return best_layers

    def discover_with_runtime_validation(
        self,
        load_config: dict,
        batch_size: int,
        max_tokens: int,
        safety_margin: float = 1.2,
        save_results: bool = True
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Discover layer split with runtime parameter validation.

        This method:
        1. Finds maximum GPU layers using binary search (load-only test)
        2. Validates with runtime parameters (batch_size, max_tokens)
        3. Finds conservative split (50% VRAM usage)

        Args:
            load_config: Base configuration for model loading
            batch_size: Batch size to validate against
            max_tokens: Max tokens to validate against
            safety_margin: Safety margin for memory estimates (default: 1.2)
            save_results: Whether to save to cache

        Returns:
            Tuple of (max_gpu_layers, min_gpu_layers)
        """
        from transformers import AutoConfig

        self.logger.info(f"Discovering layer split with runtime validation:")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Max tokens: {max_tokens}")
        self.logger.info(f"  Safety margin: {safety_margin}")

        # Get model config for memory estimation
        try:
            config = AutoConfig.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=self.trust_remote_code
            )
            model_config = {
                "hidden_size": getattr(config, "hidden_size", 4096),
                "num_attention_heads": getattr(config, "num_attention_heads", 32),
                "num_key_value_heads": getattr(config, "num_key_value_heads", None),
            }
        except Exception as e:
            self.logger.warning(f"Could not load model config: {e}, using defaults")
            model_config = {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": None,
            }

        # Detect quantization method
        quant_config = load_config.get('quantization_config')
        if quant_config:
            if hasattr(quant_config, 'load_in_4bit') and quant_config.load_in_4bit:
                quantization = '4bit'
            elif hasattr(quant_config, 'load_in_8bit') and quant_config.load_in_8bit:
                quantization = '8bit'
            else:
                quantization = 'none'
        else:
            quantization = 'none'

        # Step 1: Binary search for maximum GPU layers (load-only test)
        self.logger.info("Step 1: Finding maximum GPU layers (binary search)...")
        min_split, _ = self.find_optimal_splits(load_config, save_results=False)

        if min_split is None:
            self.logger.error("Failed to find any working layer split")
            return None, None

        max_layers = min_split.layers_on_gpu
        self.logger.info(f"✓ Load test passed with {max_layers} GPU layers")

        # Step 2: Validate with runtime parameters
        self.logger.info("Step 2: Validating with runtime parameters...")
        is_valid, estimated_peak = self._validate_split_for_runtime(
            layers_on_gpu=max_layers,
            batch_size=batch_size,
            max_tokens=max_tokens,
            model_config=model_config,
            quantization=quantization,
            safety_margin=safety_margin
        )

        if not is_valid:
            self.logger.warning(f"Runtime validation failed for {max_layers} layers, reducing...")
            max_layers = self._find_valid_split_for_runtime(
                initial_layers=max_layers,
                batch_size=batch_size,
                max_tokens=max_tokens,
                model_config=model_config,
                quantization=quantization,
                safety_margin=safety_margin
            )

        if max_layers == 0:
            self.logger.error("No layer configuration passed runtime validation")
            return None, None

        # Step 3: Find conservative minimum (50% VRAM usage)
        self.logger.info("Step 3: Finding conservative split (50% VRAM target)...")
        min_layers = self._find_conservative_split(
            max_layers=max_layers,
            target_vram_usage=0.5,
            batch_size=batch_size,
            max_tokens=max_tokens,
            model_config=model_config,
            quantization=quantization
        )

        self.logger.info(f"✓ Discovery complete:")
        self.logger.info(f"  Max performance: {max_layers}/{self.num_layers} GPU layers")
        self.logger.info(f"  Conservative: {min_layers}/{self.num_layers} GPU layers")

        return max_layers, min_layers

    def find_optimal_splits(self,
                           load_config: dict,
                           save_results: bool = True) -> Tuple[Optional[LayerSplitResult], Optional[LayerSplitResult]]:
        """
        Find minimum and maximum working layer splits using binary search.

        Args:
            load_config: Base configuration for model loading
            save_results: Whether to save successful configurations

        Returns:
            Tuple of (min_split, max_split) where:
            - min_split: Minimum CPU layers (most on GPU)
            - max_split: Maximum CPU layers (least on GPU, but > 0)
        """
        self.logger.info(f"Finding optimal layer splits for {self.model_id}")
        self.logger.info(f"Total layers: {self.num_layers}")

        if self.num_layers == 0:
            self.logger.error("Cannot determine layer count")
            return None, None

        # Binary search to find maximum GPU layers that work
        min_success = None
        left, right = 0, self.num_layers

        self.logger.info("Phase 1: Finding maximum GPU layers (minimum split)...")

        while left <= right:
            mid = (left + right) // 2
            success, gpu_mem = self.test_split(mid, load_config)

            if success:
                # This split works, try more layers on GPU
                min_success = self.builder.create_split_result(mid, gpu_mem)
                left = mid + 1
                self.logger.info(f"✓ {mid} GPU layers works, trying more...")
            else:
                # This split failed, try fewer layers on GPU
                right = mid - 1
                self.logger.info(f"✗ {mid} GPU layers failed, trying fewer...")

        if min_success is None:
            self.logger.error("No working split found!")
            return None, None

        self.logger.info(f"✓ Minimum split found: {min_success.layers_on_gpu} GPU / {min_success.layers_on_cpu} CPU")

        # Find minimum GPU layers that still use GPU (maximum split)
        # This is useful for saving GPU memory when running multiple models
        max_success = None
        if min_success.layers_on_gpu > 1:
            self.logger.info("Phase 2: Finding minimum GPU layers (maximum split)...")

            # Test with just 1 GPU layer (embeddings + last layer)
            for test_layers in range(1, min(5, min_success.layers_on_gpu)):
                success, gpu_mem = self.test_split(test_layers, load_config)
                if success:
                    max_success = self.builder.create_split_result(test_layers, gpu_mem)
                    self.logger.info(f"✓ Maximum split found: {test_layers} GPU / {self.num_layers - test_layers} CPU")
                    break
        else:
            # If min split is already 1 layer, that's also the max split
            max_success = min_success

        # Save results if requested
        if save_results and min_success and max_success:
            # Detect quantization method from config
            quant_config = load_config.get('quantization_config')
            if quant_config:
                if hasattr(quant_config, 'load_in_4bit') and quant_config.load_in_4bit:
                    quant_method = '4bit'
                elif hasattr(quant_config, 'load_in_8bit') and quant_config.load_in_8bit:
                    quant_method = '8bit'
                else:
                    quant_method = 'none'
            else:
                quant_method = 'none'

            # Estimate GPU memory from CUDA properties
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                gpu_memory_gb = 4.0  # Default

            self.config_manager.save_config(
                model_id=self.model_id,
                quant_method=quant_method,
                gpu_memory_gb=gpu_memory_gb,
                min_split=min_success,
                max_split=max_success
            )

        return min_success, max_success


def find_and_save_layer_split(model_id: str,
                              quantization_config: dict,
                              cache_dir: Optional[Path] = None,
                              trust_remote_code: bool = True) -> Optional[Tuple[LayerSplitResult, LayerSplitResult]]:
    """
    Convenience function to find and save optimal layer splits.

    Args:
        model_id: Hugging Face model ID
        quantization_config: Quantization configuration dict
        cache_dir: Cache directory
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (min_split, max_split) if successful
    """
    # Build load config
    load_config = {
        'cache_dir': str(cache_dir) if cache_dir else None,
        'trust_remote_code': trust_remote_code,
        'low_cpu_mem_usage': True
    }

    # Add quantization if specified
    if quantization_config:
        load_config['quantization_config'] = quantization_config

    # Create finder and run
    finder = LayerSplitFinder(model_id, cache_dir, trust_remote_code)
    return finder.find_optimal_splits(load_config, save_results=True)


__all__ = [
    'LayerSplitFinder',
    'find_and_save_layer_split'
]
