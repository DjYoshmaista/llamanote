#!/usr/bin/env python3
"""
Test Model Abbreviation Manager
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "model_abbreviations",
    project_root / "src" / "io" / "model_abbreviations.py"
)
model_abbrev_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_abbrev_module)
ModelAbbreviationManager = model_abbrev_module.ModelAbbreviationManager

from rich.console import Console

console = Console()


def test_configured_abbreviations():
    """Test abbreviations from configuration file"""
    console.print("\n[bold cyan]Test 1: Configured Abbreviations[/bold cyan]")

    mgr = ModelAbbreviationManager()

    # Test text models
    test_cases = [
        ("meta-llama/llama-3.2-3b", "llama_3.2_3b"),
        ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5_3b"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek_r1_1.5b"),
    ]

    for full_name, expected in test_cases:
        result = mgr.abbreviate_text_model(full_name)
        status = "✓" if result == expected else "✗"
        console.print(f"  {status} {full_name}")
        console.print(f"     → {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"

    # Test audio models
    test_cases_audio = [
        ("microsoft/speecht5_tts", "ms_speecht5"),
        ("meta-llama/AudioLlama", "metallama_audiollama"),
    ]

    for full_name, expected in test_cases_audio:
        result = mgr.abbreviate_audio_model(full_name)
        status = "✓" if result == expected else "✗"
        console.print(f"  {status} {full_name}")
        console.print(f"     → {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"

    console.print("[green]✓ Configured abbreviations test passed[/green]")


def test_fallback_patterns():
    """Test automatic fallback abbreviation patterns"""
    console.print("\n[bold cyan]Test 2: Fallback Abbreviation Patterns[/bold cyan]")

    mgr = ModelAbbreviationManager()

    # Test unknown models (not in config)
    test_cases = [
        ("huggingface/some-new-model-7b", "some_new_model_7b"),
        ("openai/gpt-4-turbo", "gpt_4_turbo"),
        ("anthropic/claude-3-sonnet", "claude_3_sonnet"),
        ("EleutherAI/gpt-neo-2.7B", "gpt_neo_2_7b"),
    ]

    for full_name, expected_pattern in test_cases:
        result = mgr.abbreviate_text_model(full_name)
        console.print(f"  ✓ {full_name}")
        console.print(f"     → {result}")
        # Check that it follows the pattern (lowercase, underscores)
        assert result.islower() or '_' in result, f"Result should be lowercase with underscores: {result}"
        assert '/' not in result, f"Result should not contain '/': {result}"
        assert '-' not in result, f"Result should not contain '-': {result}"

    console.print("[green]✓ Fallback pattern test passed[/green]")


def test_add_and_save():
    """Test adding new abbreviations"""
    console.print("\n[bold cyan]Test 3: Add and Retrieve Abbreviations[/bold cyan]")

    mgr = ModelAbbreviationManager()

    # Add new abbreviation
    mgr.add_abbreviation("text", "custom/my-model-1b", "custom_mymodel_1b")

    # Verify it's retrievable
    result = mgr.abbreviate_text_model("custom/my-model-1b")
    console.print(f"  ✓ Added: custom/my-model-1b → {result}")
    assert result == "custom_mymodel_1b"

    # Test retrieving all abbreviations
    all_abbrevs = mgr.get_all_abbreviations()
    console.print(f"  ✓ Retrieved {len(all_abbrevs['text_models'])} text model abbreviations")
    console.print(f"  ✓ Retrieved {len(all_abbrevs['audio_models'])} audio model abbreviations")

    console.print("[green]✓ Add and retrieve test passed[/green]")


def test_edge_cases():
    """Test edge cases and special characters"""
    console.print("\n[bold cyan]Test 4: Edge Cases[/bold cyan]")

    mgr = ModelAbbreviationManager()

    # Test various edge cases
    test_cases = [
        "model-with-many-hyphens-and-dots.v1.2.3",
        "UPPERCASE-MODEL-NAME",
        "model_with_underscores",
        "very/long/path/to/some/extremely/long/model/name/that/exceeds/thirty/characters",
    ]

    for model_name in test_cases:
        result = mgr.abbreviate_text_model(model_name)
        console.print(f"  ✓ {model_name}")
        console.print(f"     → {result} (len: {len(result)})")

        # Verify constraints
        assert len(result) <= 30, f"Result too long: {result}"
        assert result.islower(), f"Result should be lowercase: {result}"
        assert '/' not in result, f"Result should not contain '/': {result}"

    console.print("[green]✓ Edge cases test passed[/green]")


if __name__ == "__main__":
    console.print("[bold yellow]" + "="*60 + "[/bold yellow]")
    console.print("[bold yellow]Model Abbreviation Manager Test Suite[/bold yellow]")
    console.print("[bold yellow]" + "="*60 + "[/bold yellow]")

    try:
        test_configured_abbreviations()
        test_fallback_patterns()
        test_add_and_save()
        test_edge_cases()

        console.print("\n[bold green]" + "="*60 + "[/bold green]")
        console.print("[bold green]✅ All tests passed![/bold green]")
        console.print("[bold green]" + "="*60 + "[/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]❌ Test failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
