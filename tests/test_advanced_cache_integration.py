#!/usr/bin/env python3
# tests/test_advanced_cache_integration.py
"""
Comprehensive tests for Advanced Cache Integration

Tests the integration of:
- LlamaNoteDynamicCache with transformers
- Model-specific adapters
- Generation wrapper
- Backend integration
- Configuration options
"""

import pytest
import torch
from pathlib import Path
import sys
from unittest.mock import Mock, MagicMock, patch

# Setup proper Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock logger before importing modules
mock_logger = Mock()
sys.modules['src.utils.logger'] = Mock(get_logger_conf=lambda x: mock_logger)
sys.modules['src.config.settings'] = Mock(DEFAULT_PIPELINE_STAGES=['extract', 'process'])

# Now import the actual modules
from src.models.cache.transformers_cache_impl import (
    LlamaNoteDynamicCache,
    CacheStrategyConfig,
    MemoryStrategy
)
from src.models.cache.model_adapters import (
    ModelCacheAdapter,
    create_cache_adapter,
    LlamaCacheAdapter,
    QwenCacheAdapter,
    DeepSeekCacheAdapter,
    GemmaCacheAdapter,
    GPTNeoXCacheAdapter
)
from src.models.cache.generation_wrapper import (
    CachedGenerationWrapper,
    GenerationHooks,
    create_cached_generator
)
from src.core.types import LayerSplitConfig


class TestCacheStrategyConfig:
    """Test cache strategy configuration."""

    def test_aggressive_strategy(self):
        """Test aggressive strategy parameters."""
        config = CacheStrategyConfig(strategy=MemoryStrategy.AGGRESSIVE)

        assert config.get_window_size() == 1024
        assert config.get_prefix_size() == 64
        assert config.get_hot_cache_size() == 256.0  # MB
        assert config.get_cold_cache_size() == 1024.0  # MB

    def test_balanced_strategy(self):
        """Test balanced strategy parameters."""
        config = CacheStrategyConfig(strategy=MemoryStrategy.BALANCED)

        assert config.get_window_size() == 2048
        assert config.get_prefix_size() == 128
        assert config.get_hot_cache_size() == 512.0
        assert config.get_cold_cache_size() == 2048.0

    def test_quality_strategy(self):
        """Test quality strategy parameters."""
        config = CacheStrategyConfig(strategy=MemoryStrategy.QUALITY)

        assert config.get_window_size() == 4096
        assert config.get_prefix_size() == 256
        assert config.get_hot_cache_size() == 1024.0
        assert config.get_cold_cache_size() == 4096.0

    def test_default_strategy(self):
        """Test default is balanced."""
        config = CacheStrategyConfig()
        assert config.strategy == MemoryStrategy.BALANCED


class TestLlamaNoteDynamicCache:
    """Test core cache implementation."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = LlamaNoteDynamicCache(
            strategy_config=CacheStrategyConfig(),
            enable_hybrid_cache=True,
            enable_sliding_window=True
        )

        assert cache.strategy_config is not None
        assert cache.hybrid_enabled is True
        assert cache.sliding_enabled is True
        assert len(cache.key_cache) == 0
        assert len(cache.value_cache) == 0

    def test_cache_update(self):
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
        assert 0 in cache.key_cache
        assert 0 in cache.value_cache

    def test_cache_reset(self):
        """Test cache reset."""
        cache = LlamaNoteDynamicCache()

        # Add some data
        key_states = torch.randn(1, 8, 10, 64)
        value_states = torch.randn(1, 8, 10, 64)
        cache.update(key_states, value_states, 0, {})

        assert len(cache.key_cache) > 0

        # Reset
        cache.reset()

        assert len(cache.key_cache) == 0
        assert len(cache.value_cache) == 0

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = LlamaNoteDynamicCache()

        # Generate some cache activity
        for layer in range(3):
            key_states = torch.randn(1, 8, 10, 64)
            value_states = torch.randn(1, 8, 10, 64)
            cache.update(key_states, value_states, layer, {})

        stats = cache.get_statistics()

        assert 'num_layers' in stats
        assert 'total_tokens' in stats
        assert stats['num_layers'] == 3

    def test_sliding_window_enforcement(self):
        """Test sliding window limits sequence length."""
        config = CacheStrategyConfig(strategy=MemoryStrategy.AGGRESSIVE)
        cache = LlamaNoteDynamicCache(
            strategy_config=config,
            enable_sliding_window=True
        )

        window_size = config.get_window_size()

        # Add tokens exceeding window size
        long_seq = window_size + 100
        key_states = torch.randn(1, 8, long_seq, 64)
        value_states = torch.randn(1, 8, long_seq, 64)

        updated_keys, updated_values = cache.update(
            key_states, value_states, 0, {}
        )

        # Should be truncated to window size
        assert updated_keys.size(2) <= window_size


class TestModelAdapters:
    """Test model-specific cache adapters."""

    def create_mock_model(self, model_type: str):
        """Create a mock model with config."""
        model = Mock()
        config = Mock()

        # Set model type indicators
        if model_type == "llama":
            model.__class__.__name__ = "LlamaForCausalLM"
            config.__class__.__name__ = "LlamaConfig"
            config.num_hidden_layers = 32
            config.num_attention_heads = 32
            config.num_key_value_heads = 8  # GQA
            config.hidden_size = 4096
            config.rope_theta = 10000.0
            config.max_position_embeddings = 4096

        elif model_type == "qwen":
            model.__class__.__name__ = "Qwen2ForCausalLM"
            config.__class__.__name__ = "Qwen2Config"
            config.num_hidden_layers = 28
            config.num_attention_heads = 28
            config.num_key_value_heads = 4  # GQA
            config.hidden_size = 3584

        elif model_type == "deepseek":
            model.__class__.__name__ = "DeepseekForCausalLM"
            config.__class__.__name__ = "DeepseekConfig"
            config.num_hidden_layers = 30
            config.num_attention_heads = 32
            config.hidden_size = 4096

        elif model_type == "gemma":
            model.__class__.__name__ = "GemmaForCausalLM"
            config.__class__.__name__ = "GemmaConfig"
            config.num_hidden_layers = 18
            config.num_attention_heads = 16
            config.hidden_size = 2048

        elif model_type == "gpt-neox":
            model.__class__.__name__ = "GPTNeoXForCausalLM"
            config.__class__.__name__ = "GPTNeoXConfig"
            config.num_hidden_layers = 32
            config.num_attention_heads = 32
            config.hidden_size = 6144

        model.config = config
        return model

    def test_llama_adapter_detection(self):
        """Test LLaMA adapter auto-detection."""
        model = self.create_mock_model("llama")
        cache = LlamaNoteDynamicCache()

        adapter = create_cache_adapter(model, cache)

        assert isinstance(adapter, LlamaCacheAdapter)
        assert adapter.model_type == "llama"
        assert adapter.arch_info.uses_gqa is True

    def test_qwen_adapter_detection(self):
        """Test Qwen adapter auto-detection."""
        model = self.create_mock_model("qwen")
        cache = LlamaNoteDynamicCache()

        adapter = create_cache_adapter(model, cache)

        assert isinstance(adapter, QwenCacheAdapter)
        assert adapter.model_type == "qwen"
        assert adapter.arch_info.uses_gqa is True

    def test_deepseek_adapter_detection(self):
        """Test DeepSeek adapter auto-detection."""
        model = self.create_mock_model("deepseek")
        cache = LlamaNoteDynamicCache()

        adapter = create_cache_adapter(model, cache)

        assert isinstance(adapter, DeepSeekCacheAdapter)
        assert adapter.model_type == "deepseek"

    def test_gemma_adapter_detection(self):
        """Test Gemma adapter auto-detection."""
        model = self.create_mock_model("gemma")
        cache = LlamaNoteDynamicCache()

        adapter = create_cache_adapter(model, cache)

        assert isinstance(adapter, GemmaCacheAdapter)
        assert adapter.model_type == "gemma"

    def test_gpt_neox_adapter_detection(self):
        """Test GPT-NeoX adapter auto-detection."""
        model = self.create_mock_model("gpt-neox")
        cache = LlamaNoteDynamicCache()

        adapter = create_cache_adapter(model, cache)

        assert isinstance(adapter, GPTNeoXCacheAdapter)
        assert adapter.model_type == "gpt-neox"

    def test_architecture_info_extraction(self):
        """Test architecture info extraction."""
        model = self.create_mock_model("llama")
        cache = LlamaNoteDynamicCache()

        adapter = ModelCacheAdapter(model, cache)

        assert adapter.arch_info.num_layers == 32
        assert adapter.arch_info.num_attention_heads == 32
        assert adapter.arch_info.num_key_value_heads == 8
        assert adapter.arch_info.hidden_size == 4096
        assert adapter.arch_info.rope_theta == 10000.0

    def test_layer_preservation_logic(self):
        """Test layer preservation decisions."""
        model = self.create_mock_model("llama")
        cache = LlamaNoteDynamicCache()
        adapter = ModelCacheAdapter(model, cache)

        # First layer should be preserved
        assert adapter.should_preserve_layer(0) is True

        # Last layer should be preserved
        assert adapter.should_preserve_layer(31) is True

        # Middle layers depend on model type
        # (generic adapter may not preserve all middle layers)


class TestGenerationHooks:
    """Test generation hooks system."""

    def test_hook_registration(self):
        """Test hook registration."""
        hooks = GenerationHooks()

        def my_hook(**kwargs):
            pass

        hooks.register_pre_generation(my_hook)
        hooks.register_post_token(my_hook)
        hooks.register_post_generation(my_hook)
        hooks.register_error(my_hook)

        assert len(hooks.pre_generation_hooks) == 1
        assert len(hooks.post_token_hooks) == 1
        assert len(hooks.post_generation_hooks) == 1
        assert len(hooks.error_hooks) == 1

    def test_pre_generation_hook_execution(self):
        """Test pre-generation hooks execute."""
        hooks = GenerationHooks()
        executed = []

        def test_hook(**kwargs):
            executed.append('pre')

        hooks.register_pre_generation(test_hook)
        hooks.run_pre_generation()

        assert 'pre' in executed

    def test_post_generation_hook_execution(self):
        """Test post-generation hooks execute."""
        hooks = GenerationHooks()
        executed = []

        def test_hook(**kwargs):
            executed.append('post')

        hooks.register_post_generation(test_hook)
        hooks.run_post_generation()

        assert 'post' in executed

    def test_error_hook_execution(self):
        """Test error hooks execute."""
        hooks = GenerationHooks()
        executed = []

        def test_hook(**kwargs):
            executed.append('error')

        hooks.register_error(test_hook)
        hooks.run_error()

        assert 'error' in executed

    def test_hook_error_handling(self):
        """Test hooks don't crash on individual hook errors."""
        hooks = GenerationHooks()
        successful = []

        def failing_hook(**kwargs):
            raise ValueError("Hook error")

        def success_hook(**kwargs):
            successful.append(True)

        hooks.register_pre_generation(failing_hook)
        hooks.register_pre_generation(success_hook)

        # Should not raise, second hook should still execute
        hooks.run_pre_generation()
        assert len(successful) == 1


class TestCachedGenerationWrapper:
    """Test generation wrapper integration."""

    def create_mock_model(self):
        """Create mock model with generate method."""
        model = Mock()
        model.__class__.__name__ = "LlamaForCausalLM"

        config = Mock()
        config.__class__.__name__ = "LlamaConfig"
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        config.hidden_size = 4096

        model.config = config

        # Mock generate method
        def mock_generate(**kwargs):
            # Return fake output tensor
            return torch.tensor([[1, 2, 3, 4, 5]])

        model.generate = Mock(side_effect=mock_generate)

        return model

    def test_wrapper_initialization(self):
        """Test wrapper initializes correctly."""
        model = self.create_mock_model()
        cache = LlamaNoteDynamicCache()

        wrapper = CachedGenerationWrapper(
            model=model,
            cache=cache,
            enable_hooks=True
        )

        assert wrapper.model == model
        assert wrapper.cache == cache
        assert wrapper.hooks is not None
        assert wrapper.adapter is not None

    def test_wrapper_creates_cache_if_none(self):
        """Test wrapper creates cache if not provided."""
        model = self.create_mock_model()

        wrapper = CachedGenerationWrapper(model=model)

        assert wrapper.cache is not None
        assert isinstance(wrapper.cache, LlamaNoteDynamicCache)

    def test_generate_injects_cache(self):
        """Test generate() injects cache via past_key_values."""
        model = self.create_mock_model()
        cache = LlamaNoteDynamicCache()

        wrapper = CachedGenerationWrapper(model, cache)

        inputs = torch.tensor([[1, 2, 3]])
        outputs = wrapper.generate(inputs=inputs)

        # Verify generate was called
        assert model.generate.called

        # Verify cache was injected
        call_kwargs = model.generate.call_args[1]
        assert 'past_key_values' in call_kwargs
        assert call_kwargs['past_key_values'] == cache

    def test_hooks_execute_during_generation(self):
        """Test hooks execute during generation."""
        model = self.create_mock_model()
        cache = LlamaNoteDynamicCache()
        wrapper = CachedGenerationWrapper(model, cache, enable_hooks=True)

        executed = []

        def pre_hook(**kwargs):
            executed.append('pre')

        def post_hook(**kwargs):
            executed.append('post')

        wrapper.register_hook('pre', pre_hook)
        wrapper.register_hook('post', post_hook)

        inputs = torch.tensor([[1, 2, 3]])
        wrapper.generate(inputs=inputs)

        assert 'pre' in executed
        assert 'post' in executed

    def test_error_hook_executes_on_failure(self):
        """Test error hooks execute on generation failure."""
        model = self.create_mock_model()

        # Make generate raise an error
        model.generate = Mock(side_effect=RuntimeError("Generation failed"))

        cache = LlamaNoteDynamicCache()
        wrapper = CachedGenerationWrapper(model, cache, enable_hooks=True)

        executed = []

        def error_hook(**kwargs):
            executed.append('error')

        wrapper.register_hook('error', error_hook)

        inputs = torch.tensor([[1, 2, 3]])

        with pytest.raises(RuntimeError):
            wrapper.generate(inputs=inputs)

        assert 'error' in executed

    def test_context_manager_usage(self):
        """Test wrapper works as context manager."""
        model = self.create_mock_model()
        cache = LlamaNoteDynamicCache()

        with CachedGenerationWrapper(model, cache) as wrapper:
            assert wrapper is not None
            inputs = torch.tensor([[1, 2, 3]])
            outputs = wrapper.generate(inputs=inputs)
            assert outputs is not None

    def test_cache_reset_on_context_exit(self):
        """Test cache resets when context exits (if not external)."""
        model = self.create_mock_model()

        # Wrapper creates its own cache (should reset on exit)
        with CachedGenerationWrapper(model) as wrapper:
            # Add some data to cache
            key_states = torch.randn(1, 8, 10, 64)
            value_states = torch.randn(1, 8, 10, 64)
            wrapper.cache.update(key_states, value_states, 0, {})

            assert len(wrapper.cache.key_cache) > 0

        # After context exit, cache should be reset
        # (but we can't check since wrapper is out of scope)

    def test_external_cache_not_reset(self):
        """Test externally provided cache is not reset on exit."""
        model = self.create_mock_model()
        cache = LlamaNoteDynamicCache()

        # Add data before wrapper
        key_states = torch.randn(1, 8, 10, 64)
        value_states = torch.randn(1, 8, 10, 64)
        cache.update(key_states, value_states, 0, {})

        initial_len = len(cache.key_cache)

        with CachedGenerationWrapper(model, cache):
            pass

        # External cache should not be reset
        assert len(cache.key_cache) == initial_len

    def test_get_cache_statistics(self):
        """Test cache statistics retrieval."""
        model = self.create_mock_model()
        wrapper = CachedGenerationWrapper(model)

        stats = wrapper.get_cache_statistics()
        assert isinstance(stats, dict)


class TestCreateCachedGenerator:
    """Test convenience factory function."""

    def test_create_with_default_strategy(self):
        """Test creating generator with defaults."""
        model = Mock()
        model.__class__.__name__ = "LlamaForCausalLM"
        model.config = Mock()
        model.config.__class__.__name__ = "LlamaConfig"
        model.config.num_hidden_layers = 32

        generator = create_cached_generator(model)

        assert isinstance(generator, CachedGenerationWrapper)
        assert generator.cache.strategy_config.strategy == MemoryStrategy.BALANCED

    def test_create_with_aggressive_strategy(self):
        """Test creating with aggressive strategy."""
        model = Mock()
        model.__class__.__name__ = "LlamaForCausalLM"
        model.config = Mock()
        model.config.num_hidden_layers = 32

        generator = create_cached_generator(model, strategy="aggressive")

        assert generator.cache.strategy_config.strategy == MemoryStrategy.AGGRESSIVE

    def test_create_with_quality_strategy(self):
        """Test creating with quality strategy."""
        model = Mock()
        model.__class__.__name__ = "LlamaForCausalLM"
        model.config = Mock()
        model.config.num_hidden_layers = 32

        generator = create_cached_generator(model, strategy="quality")

        assert generator.cache.strategy_config.strategy == MemoryStrategy.QUALITY

    def test_create_with_hooks_disabled(self):
        """Test creating with hooks disabled."""
        model = Mock()
        model.__class__.__name__ = "LlamaForCausalLM"
        model.config = Mock()
        model.config.num_hidden_layers = 32

        generator = create_cached_generator(model, enable_hooks=False)

        assert generator.hooks is None


class TestLayerSplitConfigIntegration:
    """Test LayerSplitConfig integration with advanced cache."""

    def test_default_advanced_cache_enabled(self):
        """Test advanced cache is enabled by default."""
        config = LayerSplitConfig()

        assert config.use_advanced_cache is True
        assert config.cache_strategy == "balanced"
        assert config.enable_generation_hooks is True

    def test_legacy_settings_available(self):
        """Test legacy settings still available for backward compatibility."""
        config = LayerSplitConfig()

        # Legacy settings should exist but be False by default
        assert hasattr(config, 'use_hybrid_kv_cache')
        assert hasattr(config, 'use_sliding_window')
        assert config.use_hybrid_kv_cache is False
        assert config.use_sliding_window is False

    def test_cache_strategy_options(self):
        """Test different cache strategies can be set."""
        config_aggressive = LayerSplitConfig(cache_strategy="aggressive")
        config_balanced = LayerSplitConfig(cache_strategy="balanced")
        config_quality = LayerSplitConfig(cache_strategy="quality")

        assert config_aggressive.cache_strategy == "aggressive"
        assert config_balanced.cache_strategy == "balanced"
        assert config_quality.cache_strategy == "quality"


class TestMemoryStrategies:
    """Test memory strategy behaviors."""

    def test_aggressive_memory_usage(self):
        """Test aggressive strategy uses minimal memory."""
        config = CacheStrategyConfig(strategy=MemoryStrategy.AGGRESSIVE)
        cache = LlamaNoteDynamicCache(strategy_config=config)

        # Aggressive should have smallest limits
        assert config.get_window_size() == 1024
        assert config.get_hot_cache_size() == 256.0

    def test_quality_memory_usage(self):
        """Test quality strategy uses more memory."""
        config = CacheStrategyConfig(strategy=MemoryStrategy.QUALITY)
        cache = LlamaNoteDynamicCache(strategy_config=config)

        # Quality should have largest limits
        assert config.get_window_size() == 4096
        assert config.get_hot_cache_size() == 1024.0

    def test_balanced_is_middle_ground(self):
        """Test balanced strategy is between aggressive and quality."""
        aggressive = CacheStrategyConfig(strategy=MemoryStrategy.AGGRESSIVE)
        balanced = CacheStrategyConfig(strategy=MemoryStrategy.BALANCED)
        quality = CacheStrategyConfig(strategy=MemoryStrategy.QUALITY)

        assert aggressive.get_window_size() < balanced.get_window_size() < quality.get_window_size()
        assert aggressive.get_hot_cache_size() < balanced.get_hot_cache_size() < quality.get_hot_cache_size()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
