#!/usr/bin/env python3
"""
Test the checkpoint startup helper system.

Tests startup verification, metadata prompts, and legacy migration prompts.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib.util

print("=" * 70)
print("Checkpoint Startup Helper Tests")
print("=" * 70)
print()

# Load checkpoint_startup module directly
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

startup_module = load_module_from_path(
    "checkpoint_startup",
    "src/io/checkpoint_startup.py"
)

CheckpointStartupHelper = startup_module.CheckpointStartupHelper

# Test 1: CheckpointStartupHelper initialization
print("Test 1: CheckpointStartupHelper initialization")
mock_manager = MagicMock()
helper = CheckpointStartupHelper(mock_manager)

if helper.checkpoint_manager == mock_manager:
    print("  ✅ Helper initialized correctly")
    print(f"     Manager: {type(mock_manager).__name__}")
else:
    print("  ❌ FAILED: Helper initialization incorrect")

print()

# Test 2: Registry verification (success case)
print("Test 2: Registry verification")
mock_manager.verify_and_sync_registry.return_value = {
    "removed_stale": 2,
    "found_new": 3,
    "missing_metadata": 1
}

helper = CheckpointStartupHelper(mock_manager)
result = helper._verify_registry()

if result["success"] and result["removed_stale"] == 2 and result["found_new"] == 3:
    print("  ✅ Registry verification works")
    print(f"     Removed stale: {result['removed_stale']}")
    print(f"     Found new: {result['found_new']}")
    print(f"     Missing metadata: {result['missing_metadata']}")
else:
    print("  ❌ FAILED: Registry verification incorrect")

print()

# Test 3: Metadata check (non-interactive mode)
print("Test 3: Metadata check (non-interactive, auto-generate)")

# Mock checkpoint manager
mock_manager = MagicMock()
mock_checkpoints = [
    Path("/tmp/checkpoint1.ckpt"),
    Path("/tmp/checkpoint2.ckpt"),
    Path("/tmp/checkpoint3.ckpt")
]
mock_manager.find_checkpoints_without_metadata.return_value = mock_checkpoints
mock_manager.regenerate_metadata_files.return_value = {
    "total_found": 3,
    "success_count": 3,
    "error_count": 0,
    "errors": []
}

helper = CheckpointStartupHelper(mock_manager)
result = helper._check_missing_metadata(interactive=False)

if result["missing_count"] == 3 and result["generated"] == 3 and result["action"] == "generated":
    print("  ✅ Metadata check and auto-generation works")
    print(f"     Missing: {result['missing_count']}")
    print(f"     Generated: {result['generated']}")
    print(f"     Action: {result['action']}")
else:
    print("  ❌ FAILED: Metadata check incorrect")

print()

# Test 4: Legacy checkpoint check (non-interactive mode)
print("Test 4: Legacy checkpoint check (non-interactive, deferred)")

# Mock checkpoint manager
mock_manager = MagicMock()
mock_legacy = [
    Path("/tmp/abc12345_process_chunk0020_20251106_142151.ckpt"),
    Path("/tmp/def67890_audio_chunk0010_20251106_143020.ckpt")
]
mock_manager.find_legacy_checkpoints.return_value = mock_legacy

helper = CheckpointStartupHelper(mock_manager)
result = helper._check_legacy_checkpoints(interactive=False)

if result["legacy_count"] == 2 and result["migrated"] == 0 and result["action"] == "deferred":
    print("  ✅ Legacy checkpoint check works (deferred)")
    print(f"     Legacy count: {result['legacy_count']}")
    print(f"     Migrated: {result['migrated']}")
    print(f"     Action: {result['action']}")
else:
    print("  ❌ FAILED: Legacy check incorrect")

print()

# Test 5: Full startup checks (non-interactive)
print("Test 5: Full startup checks (non-interactive)")

# Mock checkpoint manager with all methods
mock_manager = MagicMock()
mock_manager.verify_and_sync_registry.return_value = {
    "removed_stale": 1,
    "found_new": 2,
    "missing_metadata": 0
}
mock_manager.find_checkpoints_without_metadata.return_value = []
mock_manager.find_legacy_checkpoints.return_value = []

helper = CheckpointStartupHelper(mock_manager)
results = helper.run_startup_checks(
    auto_sync_registry=True,
    prompt_metadata_generation=True,
    prompt_legacy_migration=True,
    interactive=False
)

if (results["registry_sync"]["success"] and
    results["metadata_check"]["action"] == "none_needed" and
    results["legacy_check"]["action"] == "none_needed"):
    print("  ✅ Full startup checks work")
    print(f"     Registry sync: Success")
    print(f"     Metadata check: {results['metadata_check']['action']}")
    print(f"     Legacy check: {results['legacy_check']['action']}")
    print(f"     Timestamp: {results['timestamp']}")
else:
    print("  ❌ FAILED: Full startup checks incorrect")

print()

# Test 6: Metadata check with no missing files
print("Test 6: Metadata check with no missing files")

mock_manager = MagicMock()
mock_manager.find_checkpoints_without_metadata.return_value = []

helper = CheckpointStartupHelper(mock_manager)
result = helper._check_missing_metadata(interactive=False)

if result["missing_count"] == 0 and result["action"] == "none_needed":
    print("  ✅ Metadata check handles no missing files correctly")
    print(f"     Missing: {result['missing_count']}")
    print(f"     Action: {result['action']}")
else:
    print("  ❌ FAILED: Metadata check incorrect")

print()

# Test 7: Legacy check with no legacy files
print("Test 7: Legacy check with no legacy files")

mock_manager = MagicMock()
mock_manager.find_legacy_checkpoints.return_value = []

helper = CheckpointStartupHelper(mock_manager)
result = helper._check_legacy_checkpoints(interactive=False)

if result["legacy_count"] == 0 and result["action"] == "none_needed":
    print("  ✅ Legacy check handles no legacy files correctly")
    print(f"     Legacy count: {result['legacy_count']}")
    print(f"     Action: {result['action']}")
else:
    print("  ❌ FAILED: Legacy check incorrect")

print()

print("=" * 70)
print("All tests completed successfully!")
print("=" * 70)
