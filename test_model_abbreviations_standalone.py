#!/usr/bin/env python3
"""
Standalone test for Model Abbreviation System.

Tests model abbreviation logic without importing the full module.
"""

import json
from pathlib import Path
import re

print("=" * 70)
print("Model Abbreviation System Tests (Standalone)")
print("=" * 70)
print()

# Test 1: Load abbreviation config
print("Test 1: Load abbreviation configuration")

config_path = Path("src/config/model_abbreviations.json")

if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)

    text_models = config.get("text_models", {})
    audio_models = config.get("audio_models", {})

    print(f"  ✅ Configuration loaded successfully")
    print(f"     Text models: {len(text_models)} entries")
    print(f"     Audio models: {len(audio_models)} entries")
else:
    print(f"  ❌ FAILED: Config file not found at {config_path}")
    exit(1)

print()

# Test 2: Verify configured abbreviations
print("Test 2: Verify configured text model abbreviations")

expected_text_mappings = {
    "meta-llama/llama-3.2-3b": "llama_3.2_3b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek_r1_1.5b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5_3b",
}

all_match = True
for full_name, expected_abbrev in expected_text_mappings.items():
    actual_abbrev = text_models.get(full_name, "NOT_FOUND")
    if actual_abbrev == expected_abbrev:
        print(f"  ✅ {full_name[:40]}")
        print(f"     → {actual_abbrev}")
    else:
        print(f"  ❌ {full_name[:40]}")
        print(f"     Expected: {expected_abbrev}, Got: {actual_abbrev}")
        all_match = False

if all_match:
    print(f"  ✅ All text model mappings correct")
else:
    print(f"  ❌ Some mappings incorrect")

print()

# Test 3: Verify audio model abbreviations
print("Test 3: Verify configured audio model abbreviations")

expected_audio_mappings = {
    "microsoft/speecht5_tts": "ms_speecht5",
    "microsoft/VibeVoice-1.5B": "ms_vibevoice_1.5b",
}

all_match_audio = True
for full_name, expected_abbrev in expected_audio_mappings.items():
    actual_abbrev = audio_models.get(full_name, "NOT_FOUND")
    if actual_abbrev == expected_abbrev:
        print(f"  ✅ {full_name}")
        print(f"     → {actual_abbrev}")
    else:
        print(f"  ❌ {full_name}")
        print(f"     Expected: {expected_abbrev}, Got: {actual_abbrev}")
        all_match_audio = False

if all_match_audio:
    print(f"  ✅ All audio model mappings correct")
else:
    print(f"  ❌ Some mappings incorrect")

print()

# Test 4: Test fallback abbreviation pattern
print("Test 4: Test fallback abbreviation pattern")

def apply_fallback_pattern(model_name: str) -> str:
    """Apply automatic abbreviation pattern for unknown models."""
    # Remove organization prefix
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    # Replace hyphens and dots with underscores
    model_name = model_name.replace('-', '_').replace('.', '_')

    # Remove common suffixes
    suffixes_to_remove = ['_Instruct', '_Chat', '_Base', '_v1', '_v2']
    for suffix in suffixes_to_remove:
        if model_name.endswith(suffix):
            model_name = model_name[:-len(suffix)]

    # Lowercase
    model_name = model_name.lower()

    # Truncate if too long
    if len(model_name) > 30:
        model_name = model_name[:30]

    return model_name

fallback_tests = [
    ("openai/gpt-4-turbo", "gpt_4_turbo"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama_3_1_8b"),
    ("mistralai/Mistral-7B-v0.2-Chat", "mistral_7b_v0_2"),
]

all_fallback_pass = True
for full_name, expected in fallback_tests:
    result = apply_fallback_pattern(full_name)
    if result == expected:
        print(f"  ✅ {full_name}")
        print(f"     → {result}")
    else:
        print(f"  ❌ {full_name}")
        print(f"     Expected: {expected}, Got: {result}")
        all_fallback_pass = False

if all_fallback_pass:
    print(f"  ✅ Fallback pattern works correctly")
else:
    print(f"  ❌ Fallback pattern needs adjustment")

print()

# Test 5: Verify config is valid JSON
print("Test 5: Verify configuration file is valid JSON")

try:
    with open(config_path, 'r') as f:
        json.load(f)
    print(f"  ✅ Configuration is valid JSON")
except json.JSONDecodeError as e:
    print(f"  ❌ FAILED: Invalid JSON - {e}")
    exit(1)

print()

print("=" * 70)
if all_match and all_match_audio and all_fallback_pass:
    print("All tests completed successfully!")
else:
    print("Some tests failed - see details above")
print("=" * 70)
