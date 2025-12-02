#!/usr/bin/env python3
"""
Unit tests for checkpoint completion percentage calculation.

Tests the weighted pipeline progress calculation method to ensure accurate
progress reporting throughout the pipeline execution.
"""

import sys
from pathlib import Path
import importlib.util

# Direct import to avoid circular import issues
def load_module_from_path(module_name, file_path):
    """Load a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load checkpoints module directly
checkpoint_module = load_module_from_path(
    "checkpoints",
    Path(__file__).parent / "src" / "io" / "checkpoints.py"
)
CheckpointManager = checkpoint_module.CheckpointManager

# Load settings module directly
settings_module = load_module_from_path(
    "settings",
    Path(__file__).parent / "src" / "config" / "settings.py"
)
DEFAULT_STAGE_WEIGHTS = settings_module.DEFAULT_STAGE_WEIGHTS


def test_completion_at_start():
    """Test completion percentage at pipeline start (0%)."""
    print("Test 1: Completion at start (0%)")

    manager = CheckpointManager()
    completion = manager.calculate_pipeline_completion(
        current_stage="extract",
        current_chunk=0,
        total_chunks=10,
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=[]
    )

    print(f"  Expected: 0%")
    print(f"  Actual: {completion}%")
    assert completion == 0, f"Expected 0%, got {completion}%"
    print("  ✅ PASSED\n")


def test_completion_at_end():
    """Test completion percentage at pipeline end (100%)."""
    print("Test 2: Completion at end (100%)")

    manager = CheckpointManager()

    # All stages complete, at end of audio stage
    all_stages = ["extract", "preprocess", "chunk", "process", "filter", "format", "save"]
    completion = manager.calculate_pipeline_completion(
        current_stage="audio",
        current_chunk=55,
        total_chunks=55,
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=all_stages
    )

    print(f"  Expected: 100%")
    print(f"  Actual: {completion}%")
    assert completion == 100, f"Expected 100%, got {completion}%"
    print("  ✅ PASSED\n")


def test_completion_mid_process_stage():
    """Test completion percentage in middle of process stage."""
    print("Test 3: Mid-process stage (chunk 20/55)")

    manager = CheckpointManager()

    # Process stage is 53.6% of total weight (45.0 / 84.0)
    # Completed: extract, preprocess, chunk = 0.5 + 0.5 + 0.3 = 1.3 weight
    # Current: process at 20/55 = 36.4% of 45.0 = 16.36 weight
    # Total progress: (1.3 + 16.36) / 84.0 = 21.02%

    completion = manager.calculate_pipeline_completion(
        current_stage="process",
        current_chunk=20,
        total_chunks=55,
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=["extract", "preprocess", "chunk"]
    )

    print(f"  Expected: ~21%")
    print(f"  Actual: {completion}%")
    assert 20 <= completion <= 22, f"Expected ~21%, got {completion}%"
    print("  ✅ PASSED\n")


def test_completion_mid_audio_stage():
    """Test completion percentage in middle of audio stage."""
    print("Test 4: Mid-audio stage (chunk 30/55)")

    manager = CheckpointManager()

    # Completed all stages except audio
    # Completed weight: 0.5 + 0.5 + 0.3 + 45.0 + 2.0 + 0.2 + 0.5 = 49.0
    # Audio progress: 30/55 = 54.5% of 35.0 = 19.09
    # Total: (49.0 + 19.09) / 84.0 = 81.06%

    completion = manager.calculate_pipeline_completion(
        current_stage="audio",
        current_chunk=30,
        total_chunks=55,
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=["extract", "preprocess", "chunk", "process", "filter", "format", "save"]
    )

    print(f"  Expected: ~81%")
    print(f"  Actual: {completion}%")
    assert 80 <= completion <= 82, f"Expected ~81%, got {completion}%"
    print("  ✅ PASSED\n")


def test_completion_stage_boundaries():
    """Test completion at stage boundaries."""
    print("Test 5: Stage boundaries")

    manager = CheckpointManager()

    # Test at end of each stage
    test_cases = [
        ("extract", ["extract"], 1),  # ~0.6%
        ("preprocess", ["extract", "preprocess"], 1),  # ~1.2%
        ("chunk", ["extract", "preprocess", "chunk"], 2),  # ~1.5%
        ("process", ["extract", "preprocess", "chunk", "process"], 54),  # ~54.2%
        ("filter", ["extract", "preprocess", "chunk", "process", "filter"], 56),  # ~56.6%
    ]

    for stage, completed, expected_pct in test_cases:
        completion = manager.calculate_pipeline_completion(
            current_stage=stage,
            current_chunk=1,  # Just finished
            total_chunks=1,
            stage_weights=DEFAULT_STAGE_WEIGHTS,
            completed_stages=completed
        )
        print(f"  After {stage}: {completion}% (expected ~{expected_pct}%)")
        # Allow some tolerance
        assert abs(completion - expected_pct) <= 2, f"Expected ~{expected_pct}%, got {completion}%"

    print("  ✅ PASSED\n")


def test_edge_cases():
    """Test edge cases (zero chunks, empty stages, etc.)."""
    print("Test 6: Edge cases")

    manager = CheckpointManager()

    # Test with 0 total chunks (should handle gracefully)
    completion = manager.calculate_pipeline_completion(
        current_stage="process",
        current_chunk=0,
        total_chunks=0,  # Edge case
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=["extract"]
    )
    print(f"  Zero chunks: {completion}%")
    assert completion >= 0 and completion <= 100, "Completion should be in valid range"

    # Test with empty completed stages
    completion = manager.calculate_pipeline_completion(
        current_stage="extract",
        current_chunk=1,
        total_chunks=1,
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=[]
    )
    print(f"  No completed stages: {completion}%")
    assert completion >= 0 and completion <= 100, "Completion should be in valid range"

    # Test with unknown stage (should use weight of 0.0)
    completion = manager.calculate_pipeline_completion(
        current_stage="unknown_stage",
        current_chunk=50,
        total_chunks=100,
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=["extract"]
    )
    print(f"  Unknown stage: {completion}%")
    assert completion >= 0 and completion <= 100, "Completion should be in valid range"

    print("  ✅ PASSED\n")


def test_weighted_stages():
    """Test that process and audio stages dominate the progress."""
    print("Test 7: Weighted stages (process and audio dominance)")

    manager = CheckpointManager()

    # Complete all fast stages (extract through save)
    # Total weight of fast stages: 0.5+0.5+0.3+2.0+0.2+0.5 = 4.0
    # This should only be ~5% of total (4.0/84.0)
    fast_stages = ["extract", "preprocess", "chunk", "filter", "format", "save"]
    completion = manager.calculate_pipeline_completion(
        current_stage="save",
        current_chunk=1,
        total_chunks=1,
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=fast_stages
    )

    print(f"  After all fast stages: {completion}%")
    assert completion < 10, f"Fast stages should be <10% of total, got {completion}%"

    # Complete process stage (should jump to ~54%)
    completion = manager.calculate_pipeline_completion(
        current_stage="process",
        current_chunk=55,
        total_chunks=55,
        stage_weights=DEFAULT_STAGE_WEIGHTS,
        completed_stages=["extract", "preprocess", "chunk"]
    )

    print(f"  After process stage: {completion}%")
    assert 53 <= completion <= 55, f"Process stage should be ~54%, got {completion}%"

    print("  ✅ PASSED\n")


def test_custom_weights():
    """Test with custom stage weights."""
    print("Test 8: Custom stage weights")

    manager = CheckpointManager()

    # Use equal weights for all stages
    equal_weights = {stage: 1.0 for stage in DEFAULT_STAGE_WEIGHTS.keys()}

    # With equal weights, completing 4 of 8 stages should be 50%
    completion = manager.calculate_pipeline_completion(
        current_stage="process",
        current_chunk=55,
        total_chunks=55,
        stage_weights=equal_weights,
        completed_stages=["extract", "preprocess", "chunk"]
    )

    print(f"  Equal weights, 4/8 stages: {completion}%")
    assert completion == 50, f"Expected 50% with equal weights, got {completion}%"

    print("  ✅ PASSED\n")


def run_all_tests():
    """Run all completion calculation tests."""
    print("=" * 70)
    print("Completion Percentage Calculation Tests")
    print("=" * 70)
    print()

    tests = [
        test_completion_at_start,
        test_completion_at_end,
        test_completion_mid_process_stage,
        test_completion_mid_audio_stage,
        test_completion_stage_boundaries,
        test_edge_cases,
        test_weighted_stages,
        test_custom_weights,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}\n")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
