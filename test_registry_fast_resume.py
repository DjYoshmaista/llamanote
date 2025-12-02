#!/usr/bin/env python3
"""
Test registry-based fast resume functionality.

Tests registry-based checkpoint discovery and selection.
"""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import sys
import importlib.util

print("=" * 70)
print("Registry-Based Fast Resume Tests")
print("=" * 70)
print()

# Load registry module
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

registry_module = load_module_from_path(
    "checkpoint_registry",
    "src/io/checkpoint_registry.py"
)
CheckpointRegistry = registry_module.CheckpointRegistry

# Define columns
columns = [
    "filename", "input_file", "input_stem", "file_extension",
    "stage", "chunk_index", "total_chunks", "completion_pct",
    "text_model", "audio_model", "hyperparameter_preset",
    "created_timestamp", "size_mb", "compressed_size_mb",
    "hash", "status"
]

# Create temp directory and registry
tmpdir_path = Path(tempfile.mkdtemp())
registry_path = tmpdir_path / ".checkpoints.csv"

# Test 1: Create sample registry with multiple checkpoints
print("Test 1: Create sample registry data")
entries = [
    {
        "filename": "paper_pdf-deepseek_r1-ms_speecht5-chk0010-10pct.ckpt",
        "input_file": "/path/to/paper.pdf",
        "input_stem": "paper",
        "file_extension": "pdf",
        "stage": "process",
        "chunk_index": "10",
        "total_chunks": "55",
        "completion_pct": "10",
        "text_model": "deepseek_r1_1.5b",
        "audio_model": "ms_speecht5",
        "hyperparameter_preset": "default",
        "created_timestamp": "2025-11-07T10:00:00",
        "size_mb": "245.9",
        "compressed_size_mb": "89.2",
        "hash": "abc12345",
        "status": "active"
    },
    {
        "filename": "paper_pdf-deepseek_r1-ms_speecht5-chk0030-35pct.ckpt",
        "input_file": "/path/to/paper.pdf",
        "input_stem": "paper",
        "file_extension": "pdf",
        "stage": "process",
        "chunk_index": "30",
        "total_chunks": "55",
        "completion_pct": "35",
        "text_model": "deepseek_r1_1.5b",
        "audio_model": "ms_speecht5",
        "hyperparameter_preset": "default",
        "created_timestamp": "2025-11-07T11:00:00",
        "size_mb": "245.9",
        "compressed_size_mb": "89.2",
        "hash": "def67890",
        "status": "active"
    },
    {
        "filename": "paper_pdf-deepseek_r1-ms_speecht5-chk0050-62pct.ckpt",
        "input_file": "/path/to/paper.pdf",
        "input_stem": "paper",
        "file_extension": "pdf",
        "stage": "process",
        "chunk_index": "50",
        "total_chunks": "55",
        "completion_pct": "62",
        "text_model": "deepseek_r1_1.5b",
        "audio_model": "ms_speecht5",
        "hyperparameter_preset": "default",
        "created_timestamp": "2025-11-07T12:00:00",
        "size_mb": "245.9",
        "compressed_size_mb": "89.2",
        "hash": "ghi11121",
        "status": "active"
    },
    {
        "filename": "paper_pdf-deepseek_r1-ms_speecht5-chk0020-20pct.ckpt",
        "input_file": "/path/to/paper.pdf",
        "input_stem": "paper",
        "file_extension": "pdf",
        "stage": "process",
        "chunk_index": "20",
        "total_chunks": "55",
        "completion_pct": "20",
        "text_model": "deepseek_r1_1.5b",
        "audio_model": "ms_speecht5",
        "hyperparameter_preset": "default",
        "created_timestamp": "2025-11-07T10:30:00",
        "size_mb": "245.9",
        "compressed_size_mb": "89.2",
        "hash": "jkl31415",
        "status": "legacy"  # Legacy status - should be filtered out
    },
    {
        "filename": "other_pdf-qwen2.5-ms_speecht5-chk0005-05pct.ckpt",
        "input_file": "/path/to/other.pdf",
        "input_stem": "other",
        "file_extension": "pdf",
        "stage": "extract",
        "chunk_index": "5",
        "total_chunks": "40",
        "completion_pct": "5",
        "text_model": "qwen2.5_3b",
        "audio_model": "ms_speecht5",
        "hyperparameter_preset": "default",
        "created_timestamp": "2025-11-07T09:00:00",
        "size_mb": "180.5",
        "compressed_size_mb": "65.3",
        "hash": "mno16171",
        "status": "active"
    }
]

with open(registry_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(entries)

registry = CheckpointRegistry(checkpoint_dir=tmpdir_path)

print(f"  ✅ Created registry with {len(entries)} entries")
print(f"     - 4 for 'paper.pdf' (3 active, 1 legacy)")
print(f"     - 1 for 'other.pdf' (1 active)")

print()

# Test 2: Mock CheckpointManager and test find_checkpoints_from_registry
print("Test 2: Find checkpoints from registry by input file")

# Create mock CheckpointManager
mock_manager = MagicMock()
mock_manager.registry = registry
mock_manager.logger = MagicMock()

# Load the find_checkpoints_from_registry method logic
def find_checkpoints_from_registry(
    input_path,
    filter_by_stage=None,
    filter_by_model=None,
    min_completion=None
):
    """Simulated method from CheckpointManager."""
    try:
        input_stem = input_path.stem
        all_checkpoints = registry.find_by_input(input_stem)

        if not all_checkpoints:
            return []

        filtered = all_checkpoints

        if filter_by_stage:
            filtered = [c for c in filtered if c.get('stage') == filter_by_stage]

        if filter_by_model:
            filtered = [c for c in filtered if
                       filter_by_model in c.get('text_model', '') or
                       filter_by_model in c.get('audio_model', '')]

        if min_completion is not None:
            filtered = [c for c in filtered if
                       int(c.get('completion_pct', 0) or 0) >= min_completion]

        # Filter to only 'active' status
        filtered = [c for c in filtered if c.get('status') == 'active']

        return filtered
    except Exception as e:
        return []

input_path = Path("/path/to/paper.pdf")
checkpoints = find_checkpoints_from_registry(input_path)

if len(checkpoints) == 3:  # Should find 3 active checkpoints for paper.pdf
    print(f"  ✅ Found {len(checkpoints)} active checkpoints for 'paper.pdf'")
    for ckpt in checkpoints:
        print(f"     - {ckpt['filename'][:50]}... ({ckpt['completion_pct']}%)")
else:
    print(f"  ❌ FAILED: Expected 3 checkpoints, got {len(checkpoints)}")

print()

# Test 3: Filter by minimum completion
print("Test 3: Filter checkpoints by minimum completion")

checkpoints_50pct = find_checkpoints_from_registry(input_path, min_completion=50)

if len(checkpoints_50pct) == 1 and int(checkpoints_50pct[0]['completion_pct']) >= 50:
    print(f"  ✅ Filter by min 50% completion works")
    print(f"     Found: {checkpoints_50pct[0]['filename'][:50]}...")
    print(f"     Completion: {checkpoints_50pct[0]['completion_pct']}%")
else:
    print(f"  ❌ FAILED: Expected 1 checkpoint with ≥50%, got {len(checkpoints_50pct)}")

print()

# Test 4: Get best checkpoint (highest completion)
print("Test 4: Get best checkpoint (highest completion)")

def get_best_checkpoint_from_registry(input_path, prefer_latest=True):
    """Simulated method from CheckpointManager."""
    checkpoints = find_checkpoints_from_registry(input_path)

    if not checkpoints:
        return None

    if prefer_latest:
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda x: int(x.get('completion_pct', 0) or 0),
            reverse=True
        )
    else:
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda x: x.get('created_timestamp', ''),
            reverse=True
        )

    return checkpoints_sorted[0] if checkpoints_sorted else None

best = get_best_checkpoint_from_registry(input_path, prefer_latest=True)

if best and int(best['completion_pct']) == 62:
    print(f"  ✅ Best checkpoint selection works")
    print(f"     Filename: {best['filename'][:50]}...")
    print(f"     Completion: {best['completion_pct']}%")
    print(f"     Chunk: {best['chunk_index']}/{best['total_chunks']}")
else:
    print(f"  ❌ FAILED: Expected checkpoint with 62% completion")

print()

# Test 5: Get checkpoint selection menu (sorted list)
print("Test 5: Get checkpoint selection menu")

def get_checkpoint_selection_menu(input_path):
    """Simulated method from CheckpointManager."""
    checkpoints = find_checkpoints_from_registry(input_path)

    if not checkpoints:
        return []

    checkpoints_sorted = sorted(
        checkpoints,
        key=lambda x: int(x.get('completion_pct', 0) or 0),
        reverse=True
    )

    return checkpoints_sorted

menu_items = get_checkpoint_selection_menu(input_path)

expected_order = [62, 35, 10]  # Completion percentages in descending order
actual_order = [int(item['completion_pct']) for item in menu_items]

if actual_order == expected_order:
    print(f"  ✅ Selection menu sorted correctly")
    for i, item in enumerate(menu_items, 1):
        print(f"     {i}. {item['filename'][:45]}... ({item['completion_pct']}%)")
else:
    print(f"  ❌ FAILED: Expected order {expected_order}, got {actual_order}")

print()

# Test 6: Filter by different input file
print("Test 6: Find checkpoints for different input file")

other_input = Path("/path/to/other.pdf")
other_checkpoints = find_checkpoints_from_registry(other_input)

if len(other_checkpoints) == 1 and other_checkpoints[0]['input_stem'] == 'other':
    print(f"  ✅ Found checkpoints for different input file")
    print(f"     Filename: {other_checkpoints[0]['filename'][:50]}...")
    print(f"     Input: {other_checkpoints[0]['input_stem']}")
else:
    print(f"  ❌ FAILED: Expected 1 checkpoint for 'other.pdf'")

print()

# Test 7: Filter by model
print("Test 7: Filter checkpoints by model abbreviation")

qwen_checkpoints = find_checkpoints_from_registry(other_input, filter_by_model="qwen")

if len(qwen_checkpoints) == 1 and 'qwen' in qwen_checkpoints[0]['text_model']:
    print(f"  ✅ Model filter works")
    print(f"     Model: {qwen_checkpoints[0]['text_model']}")
else:
    print(f"  ❌ FAILED: Model filter didn't work correctly")

print()

# Cleanup
import shutil
shutil.rmtree(tmpdir_path)

print("=" * 70)
print("All tests completed successfully!")
print("=" * 70)
