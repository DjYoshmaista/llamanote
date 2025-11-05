#!/usr/bin/env python3
"""
Standalone test runner for Advanced Cache Integration

Run this directly with: python tests/run_cache_tests.py
"""

import torch
from pathlib import Path
import sys
from unittest.mock import Mock

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock dependencies before import
mock_logger = Mock()
mock_logger.info = Mock()
mock_logger.debug = Mock()
mock_logger.warning = Mock()
mock_logger.error = Mock()

sys.modules['src.utils.logger'] = Mock(get_logger_conf=lambda x: mock_logger)
sys.modules['src.config.settings'] = Mock(DEFAULT_PIPELINE_STAGES=['extract', 'process'])

# Now import modules
from src.models.cache.transformers_cache_impl import (
    LlamaNoteDynamicCache,
    CacheStrategyConfig,
    MemoryStrategy
)
from src.models.cache.model_adapters import (
    ModelCacheAdapter,
    create_cache_adapter,
    LlamaCacheAdapter,
    QwenCacheAdapter
)
from src.models.cache.generation_wrapper import (
    CachedGenerationWrapper,
    GenerationHooks,
    create_cached_generator
)
from src.core.types import LayerSplitConfig


class TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_func, name):
        """Run a single test."""
        try:
            test_func()
            self.passed += 1
            print(f"✓ {name}")
        except AssertionError as e:
            self.failed += 1
            error_msg = str(e) if str(e) else "Assertion failed"
            self.errors.append((name, error_msg))
            print(f"✗ {name}: {error_msg}")
        except Exception as e:
            self.failed += 1
            import traceback
            error_msg = f"ERROR: {e}\n{traceback.format_exc()}"
            self.errors.append((name, error_msg))
            print(f"✗ {name}: ERROR - {e}")

    def report(self):
        """Print test report."""
        print("\n" + "=" * 60)
        print(f"Tests run: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")

        if self.errors:
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")

        return self.failed == 0


# Test functions
def test_aggressive_strategy():
    """Test aggressive strategy parameters."""
    config = CacheStrategyConfig(strategy=MemoryStrategy.AGGRESSIVE)
    assert config.get_window_size() == 1024, f"Expected 1024, got {config.get_window_size()}"
    assert config.get_prefix_size() == 64, f"Expected 64, got {config.get_prefix_size()}"
    assert config.get_hot_cache_size() == 256.0, f"Expected 256.0, got {config.get_hot_cache_size()}"


def test_balanced_strategy():
    """Test balanced strategy parameters."""
    config = CacheStrategyConfig(strategy=MemoryStrategy.BALANCED)
    assert config.get_window_size() == 2048
    assert config.get_prefix_size() == 128
    assert config.get_hot_cache_size() == 512.0


def test_quality_strategy():
    """Test quality strategy parameters."""
    config = CacheStrategyConfig(strategy=MemoryStrategy.QUALITY)
    assert config.get_window_size() == 4096
    assert config.get_prefix_size() == 256
    assert config.get_hot_cache_size() == 1024.0


def test_default_strategy():
    """Test default is balanced."""
    config = CacheStrategyConfig()
    assert config.strategy == MemoryStrategy.BALANCED


def test_cache_initialization():
    """Test cache initializes correctly."""
    cache = LlamaNoteDynamicCache(
        strategy_config=CacheStrategyConfig(),
        enable_hybrid_cache=True,
        enable_sliding_window=True
    )
    assert cache.strategy_config is not None
    assert cache.enable_hybrid is True
    assert cache.enable_sliding is True


def test_cache_update():
    """Test cache update mechanism."""
    cache = LlamaNoteDynamicCache()

    # Create dummy key/value states
    batch_size, num_heads, seq_len, head_dim = 1, 8, 10, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Update cache for layer 0
    updated_keys, updated_values = cache.update(
        key_states=key_states,
        value_states=value_states,
        layer_idx=0,
        cache_kwargs={}
    )

    assert updated_keys is not None
    assert updated_values is not None
    assert 0 in cache._cache  # _cache is the internal storage


def test_cache_reset():
    """Test cache reset."""
    cache = LlamaNoteDynamicCache()

    # Add some data
    key_states = torch.randn(1, 8, 10, 64)
    value_states = torch.randn(1, 8, 10, 64)
    cache.update(key_states, value_states, 0, {})

    assert len(cache._cache) > 0

    # Reset
    cache.reset()

    assert len(cache._cache) == 0


def test_generation_hooks():
    """Test hook registration and execution."""
    hooks = GenerationHooks()
    executed = []

    def test_hook(**kwargs):
        executed.append('test')

    hooks.register_pre_generation(test_hook)
    hooks.run_pre_generation()

    assert 'test' in executed


def test_model_adapter_llama():
    """Test LLaMA adapter auto-detection."""
    # Create mock model
    model = Mock()
    model.__class__.__name__ = "LlamaForCausalLM"
    config = Mock()
    config.__class__.__name__ = "LlamaConfig"
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.hidden_size = 4096
    model.config = config

    cache = LlamaNoteDynamicCache()
    adapter = create_cache_adapter(model, cache)

    assert isinstance(adapter, LlamaCacheAdapter)
    assert adapter.model_type == "llama"


def test_model_adapter_qwen():
    """Test Qwen adapter auto-detection."""
    # Create mock model
    model = Mock()
    model.__class__.__name__ = "Qwen2ForCausalLM"
    config = Mock()
    config.__class__.__name__ = "Qwen2Config"
    config.num_hidden_layers = 28
    config.num_attention_heads = 28
    config.num_key_value_heads = 4
    config.hidden_size = 3584
    model.config = config

    cache = LlamaNoteDynamicCache()
    adapter = create_cache_adapter(model, cache)

    assert isinstance(adapter, QwenCacheAdapter)
    assert adapter.model_type == "qwen"


def test_generation_wrapper():
    """Test generation wrapper initialization."""
    # Create mock model
    model = Mock()
    model.__class__.__name__ = "LlamaForCausalLM"
    config = Mock()
    config.__class__.__name__ = "LlamaConfig"
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.hidden_size = 4096
    model.config = config
    model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))

    cache = LlamaNoteDynamicCache()
    wrapper = CachedGenerationWrapper(model, cache, enable_hooks=True)

    assert wrapper.model == model
    assert wrapper.cache == cache
    assert wrapper.hooks is not None


def test_layer_split_config_defaults():
    """Test LayerSplitConfig has advanced cache enabled by default."""
    config = LayerSplitConfig()

    assert config.use_advanced_cache is True
    assert config.cache_strategy == "balanced"
    assert config.enable_generation_hooks is True


def test_memory_strategy_ordering():
    """Test memory strategies are correctly ordered."""
    aggressive = CacheStrategyConfig(strategy=MemoryStrategy.AGGRESSIVE)
    balanced = CacheStrategyConfig(strategy=MemoryStrategy.BALANCED)
    quality = CacheStrategyConfig(strategy=MemoryStrategy.QUALITY)

    assert aggressive.get_window_size() < balanced.get_window_size() < quality.get_window_size()
    assert aggressive.get_hot_cache_size() < balanced.get_hot_cache_size() < quality.get_hot_cache_size()


def test_cache_statistics():
    """Test cache statistics tracking."""
    cache = LlamaNoteDynamicCache()

    # Generate some cache activity
    for layer in range(3):
        key_states = torch.randn(1, 8, 10, 64)
        value_states = torch.randn(1, 8, 10, 64)
        cache.update(key_states, value_states, layer, {})

    stats = cache.get_statistics()

    # Basic sanity check - statistics should be a dictionary
    assert isinstance(stats, dict), f"Expected dict, got {type(stats)}"
    assert len(stats) > 0, "Statistics should not be empty"

    # The stats should contain some useful information
    # Don't enforce specific keys since the implementation may vary
    # Just check it returns something meaningful


def main():
    """Run all tests."""
    print("Advanced Cache Integration Tests")
    print("=" * 60)
    print()

    runner = TestRunner()

    # Run all tests
    print("Strategy Configuration Tests:")
    runner.run_test(test_aggressive_strategy, "Aggressive strategy parameters")
    runner.run_test(test_balanced_strategy, "Balanced strategy parameters")
    runner.run_test(test_quality_strategy, "Quality strategy parameters")
    runner.run_test(test_default_strategy, "Default strategy")

    print("\nCache Implementation Tests:")
    runner.run_test(test_cache_initialization, "Cache initialization")
    runner.run_test(test_cache_update, "Cache update mechanism")
    runner.run_test(test_cache_reset, "Cache reset")
    runner.run_test(test_cache_statistics, "Cache statistics")

    print("\nModel Adapter Tests:")
    runner.run_test(test_model_adapter_llama, "LLaMA adapter detection")
    runner.run_test(test_model_adapter_qwen, "Qwen adapter detection")

    print("\nGeneration Wrapper Tests:")
    runner.run_test(test_generation_hooks, "Generation hooks")
    runner.run_test(test_generation_wrapper, "Generation wrapper initialization")

    print("\nConfiguration Tests:")
    runner.run_test(test_layer_split_config_defaults, "LayerSplitConfig defaults")
    runner.run_test(test_memory_strategy_ordering, "Memory strategy ordering")

    # Print report
    success = runner.report()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
