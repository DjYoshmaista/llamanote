# llamanote/utils/auto_layer_split.py
"""
Automatic Layer Split Discovery with OOM-aware retry

Automatically discovers optimal layer splits when a new model is loaded.
Includes intelligent retry logic and user interaction for troubleshooting.
"""

import torch
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from transformers import BitsAndBytesConfig

from .layer_split_finder import LayerSplitFinder, LayerSplitResult
from .device_map_builder import LayerSplitConfigManager
from .memory_manager import CUDAMemoryManager
from .logger import get_logger_conf, ConsoleOutput

logger = get_logger_conf(__name__)


class AutoLayerSplitDiscovery:
    """
    Manages automatic discovery of optimal layer splits.

    Features:
    - Tracks which models have been tested
    - OOM-aware retry with adjusted parameters
    - User interaction for manual configuration
    - Persistent configuration storage
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize auto discovery manager.

        Args:
            config_dir: Directory for storing discovery state
        """
        if config_dir is None:
            from ..config.settings import DEFAULT_CONFIG_DIR
            config_dir = DEFAULT_CONFIG_DIR / "auto_discovery"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.config_dir / "discovery_state.json"
        self.logger = logger

        # Load or initialize state
        self.state = self._load_state()

        # Layer split config manager
        self.split_manager = LayerSplitConfigManager()

    def _load_state(self) -> Dict[str, Any]:
        """Load discovery state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load discovery state: {e}")

        return {
            "tested_models": {},  # model_id -> {quant_method -> {gpu_memory_gb -> status}}
            "version": "1.0"
        }

    def _save_state(self):
        """Save discovery state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save discovery state: {e}")

    def _get_model_key(self, model_id: str, quant_method: str, gpu_memory_gb: float) -> str:
        """Generate unique key for model configuration."""
        return f"{model_id}:{quant_method}:{gpu_memory_gb:.1f}"

    def has_been_tested(self, model_id: str, quant_method: str, gpu_memory_gb: float) -> bool:
        """Check if this model configuration has been tested."""
        model_key = self._get_model_key(model_id, quant_method, gpu_memory_gb)

        if model_id not in self.state["tested_models"]:
            return False

        if quant_method not in self.state["tested_models"][model_id]:
            return False

        if str(gpu_memory_gb) not in self.state["tested_models"][model_id][quant_method]:
            return False

        status = self.state["tested_models"][model_id][quant_method][str(gpu_memory_gb)]
        return status in ["success", "failed", "user_skipped"]

    def mark_tested(self, model_id: str, quant_method: str, gpu_memory_gb: float, status: str):
        """Mark a model configuration as tested."""
        if model_id not in self.state["tested_models"]:
            self.state["tested_models"][model_id] = {}

        if quant_method not in self.state["tested_models"][model_id]:
            self.state["tested_models"][model_id][quant_method] = {}

        self.state["tested_models"][model_id][quant_method][str(gpu_memory_gb)] = status
        self._save_state()

    def discover_with_retry(self,
                           model_id: str,
                           quant_method: str,
                           gpu_memory_gb: float,
                           quantization_config: Any,
                           max_retries: int = 3,
                           interactive: bool = True) -> Optional[Tuple[LayerSplitResult, LayerSplitResult]]:
        """
        Discover optimal layer splits with OOM-aware retry.

        Args:
            model_id: Hugging Face model ID
            quant_method: Quantization method (4bit, 8bit, none)
            gpu_memory_gb: Available GPU memory in GB
            quantization_config: BitsAndBytesConfig or similar
            max_retries: Maximum number of retry attempts
            interactive: Allow user interaction for troubleshooting

        Returns:
            Tuple of (min_split, max_split) if successful, None otherwise
        """
        # Check if already tested
        if self.has_been_tested(model_id, quant_method, gpu_memory_gb):
            self.logger.info(f"Model {model_id} already tested with {quant_method}/{gpu_memory_gb}GB")
            # Try to load existing config
            existing = self.split_manager.load_config(model_id, quant_method, gpu_memory_gb)
            if existing:
                return existing

        ConsoleOutput.header(f"Discovering Optimal Layer Split")
        ConsoleOutput.info(f"Model: {model_id}")
        ConsoleOutput.info(f"Quantization: {quant_method}")
        ConsoleOutput.info(f"GPU Memory: {gpu_memory_gb:.1f}GB")
        print()

        # Build initial load config
        load_config = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True
        }

        if quantization_config:
            load_config['quantization_config'] = quantization_config

        # Attempt discovery with retry
        attempt = 0
        last_error = None
        current_gpu_memory = gpu_memory_gb

        while attempt < max_retries:
            attempt += 1

            try:
                ConsoleOutput.info(f"Attempt {attempt}/{max_retries}: Testing with {current_gpu_memory:.1f}GB GPU memory")

                # Create finder
                finder = LayerSplitFinder(model_id)

                # Run discovery
                min_split, max_split = finder.find_optimal_splits(load_config, save_results=True)

                if min_split and max_split:
                    ConsoleOutput.success("✓ Discovery successful!")
                    ConsoleOutput.success(f"  Min split: {min_split.layers_on_gpu} GPU / {min_split.layers_on_cpu} CPU layers")
                    ConsoleOutput.success(f"  Max split: {max_split.layers_on_gpu} GPU / {max_split.layers_on_cpu} CPU layers")

                    # Mark as tested
                    self.mark_tested(model_id, quant_method, gpu_memory_gb, "success")

                    return min_split, max_split
                else:
                    raise Exception("No working split found")

            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                # Extract memory info from OOM error
                oom_memory = self._extract_oom_memory(str(e))

                ConsoleOutput.warning(f"✗ OOM error on attempt {attempt}")

                if oom_memory:
                    # Adjust GPU memory based on OOM info
                    current_gpu_memory = oom_memory * 0.9  # Use 90% of failed amount
                    ConsoleOutput.info(f"  Detected OOM at ~{oom_memory:.1f}GB, reducing to {current_gpu_memory:.1f}GB")
                else:
                    # Reduce by 20% if we can't extract info
                    current_gpu_memory *= 0.8
                    ConsoleOutput.info(f"  Reducing GPU memory to {current_gpu_memory:.1f}GB")

                # Update load config for next attempt
                if current_gpu_memory < 0.5:  # Too low, give up
                    ConsoleOutput.error("GPU memory too low, cannot continue automatic discovery")
                    break

            except Exception as e:
                last_error = e
                ConsoleOutput.warning(f"✗ Error on attempt {attempt}: {str(e)[:100]}")

        # All attempts failed
        ConsoleOutput.error("Automatic discovery failed after all retries")

        if interactive:
            return self._interactive_configuration(
                model_id, quant_method, gpu_memory_gb,
                quantization_config, last_error
            )
        else:
            self.mark_tested(model_id, quant_method, gpu_memory_gb, "failed")
            return None

    def _extract_oom_memory(self, error_msg: str) -> Optional[float]:
        """
        Extract GPU memory requirement from OOM error message.

        Args:
            error_msg: OOM error message

        Returns:
            Estimated memory in GB, or None if can't parse
        """
        import re

        # Try to find memory amounts in error message
        # Pattern: "X.XXGiB" or "XXXMB"
        patterns = [
            r'(\d+\.?\d*)\s*GiB',
            r'(\d+\.?\d*)\s*GB',
            r'(\d+)\s*MiB',
            r'(\d+)\s*MB'
        ]

        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                value = float(match.group(1))
                if 'MiB' in pattern or 'MB' in pattern:
                    value = value / 1024  # Convert to GB
                return value

        return None

    def _interactive_configuration(self,
                                   model_id: str,
                                   quant_method: str,
                                   gpu_memory_gb: float,
                                   quantization_config: Any,
                                   last_error: Optional[Exception]) -> Optional[Tuple[LayerSplitResult, LayerSplitResult]]:
        """
        Interactive manual configuration when automatic discovery fails.

        Args:
            model_id: Model ID
            quant_method: Quantization method
            gpu_memory_gb: GPU memory
            quantization_config: Quantization config
            last_error: Last error encountered

        Returns:
            Configured splits or None
        """
        print("\n" + "="*80)
        ConsoleOutput.warning("Automatic Discovery Failed - Manual Configuration Required")
        print("="*80)

        # Show current configuration
        print("\n📋 Current Configuration:")
        print(f"  Model: {model_id}")
        print(f"  Quantization: {quant_method}")
        print(f"  GPU Memory: {gpu_memory_gb:.1f}GB")

        if last_error:
            print(f"\n❌ Last Error: {str(last_error)[:200]}")

        # Show GPU info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)
            free_memory = (torch.cuda.mem_get_info()[0]) / (1024**3)
            print(f"\n🖥️  GPU Info:")
            print(f"  Device: {props.name}")
            print(f"  Total Memory: {total_memory:.1f}GB")
            print(f"  Free Memory: {free_memory:.1f}GB")

        print("\n" + "="*80)
        print("Options:")
        print("  1) Manually specify number of GPU layers to test")
        print("  2) Adjust GPU memory limit and retry")
        print("  3) Skip discovery and use device_map='auto'")
        print("  4) Cancel and return to menu")
        print("="*80)

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            return self._manual_layer_specification(model_id, quant_method, gpu_memory_gb, quantization_config)
        elif choice == "2":
            return self._manual_memory_adjustment(model_id, quant_method, gpu_memory_gb, quantization_config)
        elif choice == "3":
            ConsoleOutput.info("Skipping discovery, will use device_map='auto'")
            self.mark_tested(model_id, quant_method, gpu_memory_gb, "user_skipped")
            return None
        else:
            ConsoleOutput.info("Cancelled")
            self.mark_tested(model_id, quant_method, gpu_memory_gb, "user_cancelled")
            return None

    def _manual_layer_specification(self,
                                    model_id: str,
                                    quant_method: str,
                                    gpu_memory_gb: float,
                                    quantization_config: Any) -> Optional[Tuple[LayerSplitResult, LayerSplitResult]]:
        """Manually specify number of GPU layers."""
        from .device_map_builder import DeviceMapBuilder

        builder = DeviceMapBuilder(model_id)
        total_layers = builder.num_layers

        print(f"\n📊 Model has {total_layers} layers")
        print(f"Specify how many layers to place on GPU (0-{total_layers})")

        try:
            gpu_layers = int(input(f"GPU layers (recommended: {total_layers // 2}): ").strip())
            gpu_layers = max(0, min(gpu_layers, total_layers))
        except ValueError:
            ConsoleOutput.error("Invalid input")
            return None

        # Test this configuration
        ConsoleOutput.info(f"Testing {gpu_layers} GPU layers...")

        load_config = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
            'quantization_config': quantization_config
        }

        finder = LayerSplitFinder(model_id)
        success, gpu_mem = finder.test_split(gpu_layers, load_config)

        if success:
            # Create split results
            min_split = builder.create_split_result(gpu_layers, gpu_mem)
            max_split = builder.create_split_result(max(1, gpu_layers // 4), gpu_mem)

            # Save config
            self.split_manager.save_config(
                model_id=model_id,
                quant_method=quant_method,
                gpu_memory_gb=gpu_memory_gb,
                min_split=min_split,
                max_split=max_split
            )

            self.mark_tested(model_id, quant_method, gpu_memory_gb, "success")
            ConsoleOutput.success("✓ Configuration saved!")

            return min_split, max_split
        else:
            ConsoleOutput.error("✗ Configuration failed to load")
            return None

    def _manual_memory_adjustment(self,
                                   model_id: str,
                                   quant_method: str,
                                   gpu_memory_gb: float,
                                   quantization_config: Any) -> Optional[Tuple[LayerSplitResult, LayerSplitResult]]:
        """Manually adjust GPU memory and retry."""
        print(f"\nCurrent GPU memory limit: {gpu_memory_gb:.1f}GB")

        try:
            new_memory = float(input("New GPU memory limit (GB): ").strip())
            new_memory = max(0.5, min(new_memory, 100))  # Sanity bounds
        except ValueError:
            ConsoleOutput.error("Invalid input")
            return None

        # Retry with new memory limit
        return self.discover_with_retry(
            model_id=model_id,
            quant_method=quant_method,
            gpu_memory_gb=new_memory,
            quantization_config=quantization_config,
            max_retries=2,
            interactive=True
        )


def auto_discover_on_load(model_id: str,
                          quant_method: str,
                          gpu_memory_gb: float,
                          quantization_config: Any,
                          force_rediscover: bool = False) -> Optional[Tuple[LayerSplitResult, LayerSplitResult]]:
    """
    Automatically discover optimal layer splits when loading a model.

    This should be called before loading a model for the first time.

    Args:
        model_id: Hugging Face model ID
        quant_method: Quantization method (4bit, 8bit, none)
        gpu_memory_gb: Available GPU memory in GB
        quantization_config: BitsAndBytesConfig or similar
        force_rediscover: Force rediscovery even if already tested

    Returns:
        Tuple of (min_split, max_split) if discovery succeeded, None otherwise
    """
    discovery = AutoLayerSplitDiscovery()

    # Check if already discovered
    if not force_rediscover:
        existing = discovery.split_manager.load_config(model_id, quant_method, gpu_memory_gb)
        if existing:
            logger.info(f"Using existing layer split configuration for {model_id}")
            return existing

    # Run discovery
    return discovery.discover_with_retry(
        model_id=model_id,
        quant_method=quant_method,
        gpu_memory_gb=gpu_memory_gb,
        quantization_config=quantization_config,
        interactive=True
    )


__all__ = [
    'AutoLayerSplitDiscovery',
    'auto_discover_on_load'
]
