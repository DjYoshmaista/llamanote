#!/usr/bin/env python3
"""
Simple unit tests for completion percentage calculation.
Tests the calculation logic directly without complex imports.
"""

# Define the calculation function inline for testing
def calculate_pipeline_completion(
    current_stage: str,
    current_chunk: int,
    total_chunks: int,
    stage_weights: dict,
    completed_stages: list
) -> int:
    """Calculate overall pipeline completion percentage using weighted stages."""
    # Sum weight of completed stages
    completed_weight = sum(stage_weights.get(s, 0.0) for s in completed_stages)

    # Add partial weight of current stage
    current_stage_weight = stage_weights.get(current_stage, 0.0)
    if total_chunks > 0:
        stage_progress = current_chunk / total_chunks
    else:
        stage_progress = 0.0
    current_weight = current_stage_weight * stage_progress

    # Calculate percentage
    total_weight = sum(stage_weights.values())
    if total_weight > 0:
        completion_pct = (completed_weight + current_weight) / total_weight * 100
    else:
        completion_pct = 0.0

    # Round to nearest integer and clamp to 0-100 range
    return max(0, min(100, round(completion_pct)))


# Default stage weights from settings
DEFAULT_STAGE_WEIGHTS = {
    "extract": 0.5,
    "preprocess": 0.5,
    "chunk": 0.3,
    "process": 45.0,
    "filter": 2.0,
    "format": 0.2,
    "save": 0.5,
    "audio": 35.0,
}


def run_tests():
    """Run all tests."""
    print("=" * 70)
    print("Completion Percentage Calculation Tests")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    # Test 1: Start (0%)
    print("Test 1: Completion at start (0%)")
    result = calculate_pipeline_completion(
        "extract", 0, 10, DEFAULT_STAGE_WEIGHTS, []
    )
    print(f"  Expected: 0%, Actual: {result}%")
    if result == 0:
        print("  ✅ PASSED\n")
        passed += 1
    else:
        print(f"  ❌ FAILED\n")
        failed += 1

    # Test 2: End (100%)
    print("Test 2: Completion at end (100%)")
    all_stages = ["extract", "preprocess", "chunk", "process", "filter", "format", "save"]
    result = calculate_pipeline_completion(
        "audio", 55, 55, DEFAULT_STAGE_WEIGHTS, all_stages
    )
    print(f"  Expected: 100%, Actual: {result}%")
    if result == 100:
        print("  ✅ PASSED\n")
        passed += 1
    else:
        print(f"  ❌ FAILED\n")
        failed += 1

    # Test 3: Mid-process (chunk 20/55) - should be ~21%
    print("Test 3: Mid-process stage (chunk 20/55)")
    result = calculate_pipeline_completion(
        "process", 20, 55, DEFAULT_STAGE_WEIGHTS,
        ["extract", "preprocess", "chunk"]
    )
    print(f"  Expected: ~21%, Actual: {result}%")
    if 20 <= result <= 22:
        print("  ✅ PASSED\n")
        passed += 1
    else:
        print(f"  ❌ FAILED\n")
        failed += 1

    # Test 4: Mid-audio (chunk 30/55) - should be ~81%
    print("Test 4: Mid-audio stage (chunk 30/55)")
    result = calculate_pipeline_completion(
        "audio", 30, 55, DEFAULT_STAGE_WEIGHTS,
        ["extract", "preprocess", "chunk", "process", "filter", "format", "save"]
    )
    print(f"  Expected: ~81%, Actual: {result}%")
    if 80 <= result <= 82:
        print("  ✅ PASSED\n")
        passed += 1
    else:
        print(f"  ❌ FAILED\n")
        failed += 1

    # Test 5: Zero chunks edge case
    print("Test 5: Edge case - zero total chunks")
    result = calculate_pipeline_completion(
        "process", 0, 0, DEFAULT_STAGE_WEIGHTS, ["extract"]
    )
    print(f"  Expected: 0-100 range, Actual: {result}%")
    if 0 <= result <= 100:
        print("  ✅ PASSED\n")
        passed += 1
    else:
        print(f"  ❌ FAILED\n")
        failed += 1

    # Test 6: Weighted stage dominance
    print("Test 6: Fast stages should be <10% of total")
    fast_stages = ["extract", "preprocess", "chunk", "filter", "format", "save"]
    result = calculate_pipeline_completion(
        "save", 1, 1, DEFAULT_STAGE_WEIGHTS, fast_stages
    )
    print(f"  Expected: <10%, Actual: {result}%")
    if result < 10:
        print("  ✅ PASSED\n")
        passed += 1
    else:
        print(f"  ❌ FAILED\n")
        failed += 1

    # Test 7: Process stage completion should be ~54%
    print("Test 7: After process stage (~54%)")
    result = calculate_pipeline_completion(
        "process", 55, 55, DEFAULT_STAGE_WEIGHTS,
        ["extract", "preprocess", "chunk"]
    )
    print(f"  Expected: ~54%, Actual: {result}%")
    if 53 <= result <= 55:
        print("  ✅ PASSED\n")
        passed += 1
    else:
        print(f"  ❌ FAILED\n")
        failed += 1

    # Test 8: Custom equal weights
    print("Test 8: Custom equal weights (4/8 stages = 50%)")
    equal_weights = {stage: 1.0 for stage in DEFAULT_STAGE_WEIGHTS.keys()}
    result = calculate_pipeline_completion(
        "process", 55, 55, equal_weights,
        ["extract", "preprocess", "chunk"]
    )
    print(f"  Expected: 50%, Actual: {result}%")
    if result == 50:
        print("  ✅ PASSED\n")
        passed += 1
    else:
        print(f"  ❌ FAILED\n")
        failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
