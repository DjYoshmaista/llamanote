#!/usr/bin/env python3
"""
Test script for ChatTemplateManager

This script tests the chat template manager module to ensure it works correctly
before integration into the main pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Direct import to avoid circular dependencies
from src.utils.chat_template_manager import ChatTemplateManager, auto_detect_and_apply_template


def test_template_detection():
    """Test automatic template detection for various models."""
    print("=" * 70)
    print("TEST 1: Template Detection")
    print("=" * 70)

    manager = ChatTemplateManager()

    test_models = [
        "TinyLlama/TinyLlama_v1.1",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen2.5-3B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "unknown-model/test",
    ]

    for model_name in test_models:
        template = manager.get_template_for_model(model_name)
        if template:
            print(f"✅ {model_name}: {template.name}")
        else:
            print(f"❌ {model_name}: No template found")

    print()


def test_template_listing():
    """Test listing available templates."""
    print("=" * 70)
    print("TEST 2: Available Templates")
    print("=" * 70)

    manager = ChatTemplateManager()
    templates = manager.list_templates()

    print(f"Found {len(templates)} templates:")
    for name in templates:
        info = manager.get_template_info(name)
        if info:
            print(f"  - {name}: {info['description']}")

    print()


def test_template_application():
    """Test template application with a mock tokenizer."""
    print("=" * 70)
    print("TEST 3: Template Application (Mock)")
    print("=" * 70)

    class MockTokenizer:
        """Mock tokenizer for testing."""
        def __init__(self):
            self.chat_template = None
            self.bos_token = None
            self.eos_token = None
            self.pad_token = None
            self.unk_token = None
            self.padding_side = None

    manager = ChatTemplateManager()
    tokenizer = MockTokenizer()

    # Test applying TinyLlama template
    result = manager.apply_template(tokenizer, model_name="TinyLlama/TinyLlama_v1.1")

    if result:
        print(f"✅ Template applied successfully")
        print(f"   Template set: {tokenizer.chat_template is not None}")
        print(f"   Padding side: {tokenizer.padding_side}")
    else:
        print(f"❌ Template application failed")

    print()


def test_template_format():
    """Test template formatting (requires transformers)."""
    print("=" * 70)
    print("TEST 4: Template Formatting")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer

        # Create a simple tokenizer
        print("Creating tokenizer for TinyLlama...")
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama_v1.1",
            use_fast=True,
        )

        # Apply template
        manager = ChatTemplateManager()
        manager.apply_template(tokenizer, model_name="TinyLlama/TinyLlama_v1.1", force=True)

        # Test formatting
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print("✅ Template formatting test:")
        print("-" * 70)
        print(formatted)
        print("-" * 70)

    except ImportError:
        print("⚠️  Transformers not installed, skipping formatting test")
    except Exception as e:
        print(f"❌ Formatting test failed: {e}")

    print()


def main():
    """Run all tests."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Chat Template Manager Test Suite" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    test_template_detection()
    test_template_listing()
    test_template_application()
    test_template_format()

    print("=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
