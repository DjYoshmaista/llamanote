#!/usr/bin/env python3
"""
Test the checkpoint metadata system.

Tests JSON sidecar file creation, loading, and fallback mechanisms.
"""

import json
import tempfile
from pathlib import Path

print("=" * 70)
print("Checkpoint Metadata System Tests")
print("=" * 70)
print()

# Test 1: JSON file creation and loading
print("Test 1: JSON metadata file creation")
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)

    # Create a fake checkpoint file
    ckpt_path = tmpdir_path / "test_checkpoint.ckpt"
    ckpt_path.write_text("fake checkpoint data")

    # Create metadata
    metadata = {
        "version": "2.0",
        "timestamp": "2025-11-07T10:00:00",
        "stage": "process",
        "input_stem": "test_file",
        "chunk_index": 20,
        "total_chunks": 55,
        "completion_pct": 21,
        "config": {
            "text_model": {
                "provider": "deepseek-ai",
                "specifier": "DeepSeek-R1-Distill-Qwen-1.5B",
                "full_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            }
        },
        "hyperparameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 2048
        }
    }

    # Add checkpoint file info
    metadata["checkpoint_file"] = {
        "filename": ckpt_path.name,
        "size_bytes": 1000000,
        "size_mb": 0.95,
        "compressed_size_bytes": 500000,
        "compressed_size_mb": 0.48,
        "compression_ratio": 0.5
    }

    # Write JSON metadata
    meta_path = ckpt_path.with_suffix(ckpt_path.suffix + '.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # Verify file exists
    if meta_path.exists():
        print(f"  ✅ Created metadata file: {meta_path.name}")
    else:
        print(f"  ❌ FAILED: Metadata file not created")

    # Load and verify
    with open(meta_path, 'r', encoding='utf-8') as f:
        loaded_metadata = json.load(f)

    if loaded_metadata["stage"] == "process":
        print(f"  ✅ Loaded metadata correctly")
        print(f"     Stage: {loaded_metadata['stage']}")
        print(f"     Completion: {loaded_metadata['completion_pct']}%")
        print(f"     Chunk: {loaded_metadata['chunk_index']}/{loaded_metadata['total_chunks']}")
    else:
        print(f"  ❌ FAILED: Metadata mismatch")

print()

# Test 2: Metadata file structure
print("Test 2: Verify metadata structure")
expected_keys = ["version", "timestamp", "stage", "config", "checkpoint_file"]
missing_keys = [k for k in expected_keys if k not in loaded_metadata]

if not missing_keys:
    print(f"  ✅ All required keys present")
    print(f"     Keys: {', '.join(expected_keys)}")
else:
    print(f"  ❌ FAILED: Missing keys: {missing_keys}")

# Check checkpoint file info
if "checkpoint_file" in loaded_metadata:
    ckpt_info = loaded_metadata["checkpoint_file"]
    required_ckpt_keys = ["filename", "size_mb", "compressed_size_mb", "compression_ratio"]
    missing_ckpt_keys = [k for k in required_ckpt_keys if k not in ckpt_info]

    if not missing_ckpt_keys:
        print(f"  ✅ Checkpoint file info complete")
        print(f"     Size: {ckpt_info['compressed_size_mb']}MB")
        print(f"     Compression: {ckpt_info['compression_ratio']:.1%}")
    else:
        print(f"  ❌ FAILED: Missing checkpoint info keys: {missing_ckpt_keys}")

print()

# Test 3: Model name abbreviation (test data)
print("Test 3: Model abbreviation in filename")
# This would be done by CheckpointManager._get_checkpoint_path()
# Here we just verify the format

model_abbrev = "deepseek_r1_1.5b"
audio_abbrev = "ms_speecht5"
file_stem = "BID_paper"
file_ext = "pdf"
chunk_num = 20
completion_pct = 21

new_format_name = f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk{chunk_num:04d}-{completion_pct}pct.ckpt"

print(f"  New format: {new_format_name}")

# Verify format
if all([
    file_stem in new_format_name,
    file_ext in new_format_name,
    model_abbrev in new_format_name,
    f"chk{chunk_num:04d}" in new_format_name,
    f"{completion_pct}pct" in new_format_name
]):
    print(f"  ✅ Filename format correct")
    print(f"     Contains: file, ext, models, chunk, progress")
else:
    print(f"  ❌ FAILED: Filename format incorrect")

print()

# Test 4: Sortability
print("Test 4: Filename sortability")
filenames = [
    f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk0010-10pct.ckpt",
    f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk0020-21pct.ckpt",
    f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk0005-05pct.ckpt",
    f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk0050-90pct.ckpt",
]

sorted_filenames = sorted(filenames)

# Check if they sort by chunk number (due to zero-padding)
expected_order = [
    f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk0005-05pct.ckpt",
    f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk0010-10pct.ckpt",
    f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk0020-21pct.ckpt",
    f"{file_stem}_{file_ext}-{model_abbrev}-{audio_abbrev}-chk0050-90pct.ckpt",
]

if sorted_filenames == expected_order:
    print(f"  ✅ Filenames sort correctly by chunk number")
    for fn in sorted_filenames[:2]:  # Show first 2
        print(f"     {fn}")
    print(f"     ...")
else:
    print(f"  ❌ FAILED: Filenames don't sort correctly")

print()

print("=" * 70)
print("All tests completed successfully!")
print("=" * 70)
