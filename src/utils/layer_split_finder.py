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
