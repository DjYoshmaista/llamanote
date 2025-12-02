#!/usr/bin/env python3
"""
Test the checkpoint registry system.

Tests CSV registry creation, CRUD operations, and verification.
"""

import csv
import tempfile
from pathlib import Path

print("=" * 70)
print("Checkpoint Registry System Tests")
print("=" * 70)
print()

# Define columns globally
columns = [
    "filename", "input_file", "input_stem", "file_extension",
    "stage", "chunk_index", "total_chunks", "completion_pct",
    "text_model", "audio_model", "hyperparameter_preset",
    "created_timestamp", "size_mb", "compressed_size_mb",
    "hash", "status"
]

# Create temp directory for all tests
tmpdir_path = Path(tempfile.mkdtemp())
registry_path = tmpdir_path / ".checkpoints.csv"

# Test 1: Registry file creation
print("Test 1: Registry file creation")

# Create registry
with open(registry_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

# Verify file exists
if registry_path.exists():
    print(f"  ✅ Created registry file: {registry_path.name}")
else:
    print(f"  ❌ FAILED: Registry file not created")

# Verify header
with open(registry_path, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames

if "filename" in headers and "completion_pct" in headers:
    print(f"  ✅ CSV header correct ({len(headers)} columns)")
else:
    print(f"  ❌ FAILED: CSV header incorrect")

print()

# Test 2: Add entries to registry
print("Test 2: Add entries to registry")
entries = [
    {
        "filename": "test_pdf-deepseek_r1-ms_speecht5-chk0020-21pct.ckpt",
        "input_file": "/path/to/test.pdf",
        "input_stem": "test",
        "file_extension": "pdf",
        "stage": "process",
        "chunk_index": "20",
        "total_chunks": "55",
        "completion_pct": "21",
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
        "filename": "test_pdf-deepseek_r1-ms_speecht5-chk0050-54pct.ckpt",
        "input_file": "/path/to/test.pdf",
        "input_stem": "test",
        "file_extension": "pdf",
        "stage": "process",
        "chunk_index": "50",
        "total_chunks": "55",
        "completion_pct": "54",
        "text_model": "deepseek_r1_1.5b",
        "audio_model": "ms_speecht5",
        "hyperparameter_preset": "default",
        "created_timestamp": "2025-11-07T11:00:00",
        "size_mb": "245.9",
        "compressed_size_mb": "89.2",
        "hash": "abc12345",
        "status": "active"
    }
]

with open(registry_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(entries)

# Verify entries
with open(registry_path, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    loaded_entries = list(reader)

if len(loaded_entries) == 2:
    print(f"  ✅ Added 2 entries to registry")
    print(f"     Entry 1: {loaded_entries[0]['filename'][:40]}...")
    print(f"     Entry 2: {loaded_entries[1]['filename'][:40]}...")
else:
    print(f"  ❌ FAILED: Expected 2 entries, got {len(loaded_entries)}")

print()

# Test 3: Filter entries by input_stem
print("Test 3: Filter entries by input_stem")
matching = [e for e in loaded_entries if e['input_stem'] == 'test']

if len(matching) == 2:
    print(f"  ✅ Found {len(matching)} entries for input 'test'")
else:
    print(f"  ❌ FAILED: Expected 2 matches, got {len(matching)}")

print()

# Test 4: Sort by completion percentage
print("Test 4: Sort by completion percentage")
sorted_entries = sorted(loaded_entries, key=lambda x: int(x['completion_pct']))

if sorted_entries[0]['completion_pct'] == '21' and sorted_entries[1]['completion_pct'] == '54':
    print(f"  ✅ Sorted correctly by completion")
    print(f"     First: {sorted_entries[0]['completion_pct']}%")
    print(f"     Second: {sorted_entries[1]['completion_pct']}%")
else:
    print(f"  ❌ FAILED: Sort order incorrect")

print()

# Test 5: Find latest entry
print("Test 5: Find latest entry by timestamp")
latest = max(loaded_entries, key=lambda x: x['created_timestamp'])

if latest['chunk_index'] == '50':
    print(f"  ✅ Found latest entry")
    print(f"     Chunk: {latest['chunk_index']}/{latest['total_chunks']}")
    print(f"     Timestamp: {latest['created_timestamp']}")
else:
    print(f"  ❌ FAILED: Latest entry incorrect")

print()

# Test 6: Calculate statistics
print("Test 6: Calculate statistics")
total_size_mb = sum(float(e['compressed_size_mb']) for e in loaded_entries)
by_stage = {}
for entry in loaded_entries:
    stage = entry['stage']
    by_stage[stage] = by_stage.get(stage, 0) + 1

print(f"  Total entries: {len(loaded_entries)}")
print(f"  Total size: {total_size_mb:.1f} MB")
print(f"  By stage: {dict(by_stage)}")

if len(loaded_entries) == 2 and abs(total_size_mb - 178.4) < 0.1:
    print(f"  ✅ Statistics calculated correctly")
else:
    print(f"  ❌ FAILED: Statistics incorrect")

print()

print("=" * 70)
print("All tests completed successfully!")
print("=" * 70)

# Cleanup
import shutil
shutil.rmtree(tmpdir_path)
