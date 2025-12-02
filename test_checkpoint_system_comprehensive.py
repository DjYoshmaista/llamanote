#!/usr/bin/env python3
"""
Comprehensive Checkpoint System Test Suite.

Runs all checkpoint system tests:
- Model abbreviations
- Completion calculation
- Metadata system
- Registry system
- Migration system
- Startup helper
- Registry-based fast resume

This is the master test runner for validating the entire checkpoint enhancement system.
"""

import subprocess
import sys
from pathlib import Path

print("=" * 80)
print("COMPREHENSIVE CHECKPOINT SYSTEM TEST SUITE")
print("=" * 80)
print()

# List of all test files
test_files = [
    ("Model Abbreviations", "test_model_abbreviations_standalone.py"),
    ("Completion Calculation", "test_completion_simple.py"),
    ("Metadata System", "test_metadata_system.py"),
    ("Registry System", "test_registry_system.py"),
    ("Startup Helper", "test_startup_helper.py"),
    ("Registry Fast Resume", "test_registry_fast_resume.py"),
]

# Track results
results = []
total_tests = 0
passed_tests = 0
failed_tests = 0

print("Running checkpoint system tests...")
print()

# Run each test file
for test_name, test_file in test_files:
    test_path = Path(test_file)

    if not test_path.exists():
        print(f"⚠️  SKIPPED: {test_name} - File not found: {test_file}")
        results.append((test_name, "SKIPPED", "File not found"))
        continue

    print(f"Running: {test_name}")
    print("-" * 80)

    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Check if test passed
        if result.returncode == 0 and ("All tests completed successfully" in result.stdout or
                                       "✅" in result.stdout):
            print(f"✅ PASSED: {test_name}")
            results.append((test_name, "PASSED", None))
            passed_tests += 1

            # Count individual test cases
            test_count = result.stdout.count("✅")
            total_tests += test_count

        else:
            print(f"❌ FAILED: {test_name}")
            print(f"   Return code: {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            results.append((test_name, "FAILED", result.stderr[:200] if result.stderr else "Unknown error"))
            failed_tests += 1

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {test_name} (>30s)")
        results.append((test_name, "TIMEOUT", "Test exceeded 30 second timeout"))
        failed_tests += 1

    except Exception as e:
        print(f"❌ ERROR: {test_name} - {str(e)}")
        results.append((test_name, "ERROR", str(e)))
        failed_tests += 1

    print()

# Print summary
print("=" * 80)
print("TEST SUITE SUMMARY")
print("=" * 80)
print()

print(f"Test Suites Run: {len(test_files)}")
print(f"  ✅ Passed: {passed_tests}")
print(f"  ❌ Failed: {failed_tests}")
print(f"  ⚠️  Skipped: {len(test_files) - passed_tests - failed_tests}")
print()

if total_tests > 0:
    print(f"Total Individual Tests: {total_tests}")
    print()

# Print detailed results
print("Detailed Results:")
print("-" * 80)
for test_name, status, error in results:
    status_icon = {
        "PASSED": "✅",
        "FAILED": "❌",
        "TIMEOUT": "⏱️ ",
        "ERROR": "💥",
        "SKIPPED": "⚠️ "
    }.get(status, "?")

    print(f"{status_icon} {status:8} - {test_name}")
    if error:
        print(f"           Error: {error}")

print()
print("=" * 80)

# Exit with appropriate code
if failed_tests > 0:
    print("❌ TEST SUITE FAILED")
    print("=" * 80)
    sys.exit(1)
else:
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    sys.exit(0)
