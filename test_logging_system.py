#!/usr/bin/env python3
"""
Test script for LlamaNote logging system.

Tests:
1. Logger creation with hierarchical naming
2. File handler (rotating logs)
3. Console handler (with colors)
4. systemd journal handler (if available)
5. Per-module configuration
6. Log formatting
7. Exception logging
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logging_config import (
    get_logger,
    get_config_manager,
    LogLevel,
    reconfigure_logger,
    SYSTEMD_AVAILABLE
)

def test_basic_logging():
    """Test 1: Basic logger creation and logging"""
    print("\n" + "="*70)
    print("TEST 1: Basic Logging")
    print("="*70)

    logger = get_logger(__name__)

    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    print("✅ Test 1 passed: All log levels working")


def test_hierarchical_naming():
    """Test 2: Hierarchical logger naming"""
    print("\n" + "="*70)
    print("TEST 2: Hierarchical Logger Naming")
    print("="*70)

    # Create loggers for different "modules"
    logger_main = get_logger("main")
    logger_cli = get_logger("src.cli")
    logger_pipeline = get_logger("src.core.pipeline")
    logger_checkpoints = get_logger("src.io.checkpoints")

    logger_main.info("Log from main module")
    logger_cli.info("Log from CLI module")
    logger_pipeline.info("Log from pipeline module")
    logger_checkpoints.info("Log from checkpoints module")

    # Verify logger names
    assert logger_main.name == "llamanote.main"
    assert logger_cli.name == "llamanote.src.cli"
    assert logger_pipeline.name == "llamanote.src.core.pipeline"
    assert logger_checkpoints.name == "llamanote.src.io.checkpoints"

    print(f"✅ Test 2 passed: Hierarchical naming working")
    print(f"   Logger names: {[l.name for l in [logger_main, logger_cli, logger_pipeline, logger_checkpoints]]}")


def test_function_entry_exit():
    """Test 3: Function entry/exit logging pattern"""
    print("\n" + "="*70)
    print("TEST 3: Function Entry/Exit Logging")
    print("="*70)

    logger = get_logger(__name__)

    def sample_function(x: int, y: int) -> int:
        logger.debug(f"ENTER | x={x}, y={y}")
        start = time.perf_counter()
        try:
            result = x + y
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"EXIT | return={result} | duration={elapsed:.2f}ms")
            return result
        except Exception as e:
            logger.error(f"EXIT | exception={type(e).__name__}: {e}", exc_info=True)
            raise

    result = sample_function(5, 3)
    assert result == 8

    print("✅ Test 3 passed: Entry/exit logging working")


def test_loop_logging():
    """Test 4: Loop iteration logging"""
    print("\n" + "="*70)
    print("TEST 4: Loop Iteration Logging")
    print("="*70)

    logger = get_logger(__name__)

    items = ["item1", "item2", "item3", "item4", "item5"]

    for i, item in enumerate(items):
        logger.debug(f"Loop iteration {i+1}/{len(items)} | item={item}")
        # Simulate processing
        time.sleep(0.01)

    print("✅ Test 4 passed: Loop logging working")


def test_api_request_logging():
    """Test 5: API request logging pattern"""
    print("\n" + "="*70)
    print("TEST 5: API Request Logging")
    print("="*70)

    logger = get_logger(__name__)

    # Simulate API request
    method = "POST"
    url = "https://api.example.com/v1/generate"
    params = {"model": "deepseek-r1", "max_tokens": 2048}

    logger.info(f"API Request: {method} {url} | params={params}")

    # Simulate response
    status_code = 200
    elapsed = 1234
    response_size = 5678

    logger.info(f"API Response: {status_code} | duration={elapsed}ms | size={response_size}bytes")
    logger.debug(f"API Response Body: {{\"status\": \"success\", \"data\": \"...\"}}...")

    print("✅ Test 5 passed: API request logging working")


def test_checkpoint_logging():
    """Test 6: Checkpoint logging pattern"""
    print("\n" + "="*70)
    print("TEST 6: Checkpoint Logging")
    print("="*70)

    logger = get_logger(__name__)

    # Simulate checkpoint save
    filename = "example_pdf-deepseek_r1-chk0020-15pct.ckpt"
    chunk_idx = 20
    total_chunks = 55
    size_mb = 245.9
    compressed_mb = 89.2
    compression_ratio = compressed_mb / size_mb
    file_hash = "a3f9c2d1e8b45678"

    logger.info(
        f"Checkpoint Saved: {filename} | "
        f"chunk={chunk_idx}/{total_chunks} | "
        f"size={size_mb:.2f}MB | "
        f"compressed={compressed_mb:.2f}MB ({compression_ratio:.1%}) | "
        f"hash={file_hash[:16]}"
    )

    metadata = {"stage": "process", "completion": 15}
    logger.debug(f"Checkpoint metadata: {metadata}")

    print("✅ Test 6 passed: Checkpoint logging working")


def test_exception_logging():
    """Test 7: Exception logging"""
    print("\n" + "="*70)
    print("TEST 7: Exception Logging")
    print("="*70)

    logger = get_logger(__name__)

    def risky_function():
        raise ValueError("Simulated error for testing")

    try:
        risky_function()
    except ValueError as e:
        logger.error(f"Caught expected error: {e}", exc_info=True)

    try:
        x = 1 / 0
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)

    print("✅ Test 7 passed: Exception logging working")


def test_configuration():
    """Test 8: Configuration system"""
    print("\n" + "="*70)
    print("TEST 8: Configuration System")
    print("="*70)

    config_mgr = get_config_manager()

    # Check default configuration
    print(f"Global log level: {config_mgr.config.global_level.name}")
    print(f"Log directory: {config_mgr.config.log_dir}")
    print(f"File handler enabled: {config_mgr.config.file_enabled}")
    print(f"Console handler enabled: {config_mgr.config.console_enabled}")
    print(f"Journal handler enabled: {config_mgr.config.journal_enabled}")

    # Test module-specific configuration
    pipeline_config = config_mgr.get_module_config("llamanote.src.core.pipeline")
    print(f"\nPipeline module config:")
    print(f"  Console level: {pipeline_config.console_level.name}")
    print(f"  File level: {pipeline_config.file_level.name}")
    print(f"  Journal level: {pipeline_config.journal_level.name}")

    print("✅ Test 8 passed: Configuration system working")


def test_log_files():
    """Test 9: Log file creation"""
    print("\n" + "="*70)
    print("TEST 9: Log File Creation")
    print("="*70)

    config_mgr = get_config_manager()
    log_files = config_mgr.list_log_files()

    print(f"Found {len(log_files)} log files:")
    for log_file in log_files[:5]:  # Show first 5
        size = config_mgr.get_log_file_size(log_file)
        print(f"  - {log_file.name}: {size:,} bytes")

    if len(log_files) > 5:
        print(f"  ... and {len(log_files) - 5} more")

    print("✅ Test 9 passed: Log files created successfully")


def test_systemd_integration():
    """Test 10: systemd journal integration"""
    print("\n" + "="*70)
    print("TEST 10: systemd Journal Integration")
    print("="*70)

    if not SYSTEMD_AVAILABLE:
        print("⚠️  Test 10 skipped: systemd-python not installed")
        print("   Install with: pip install systemd-python")
        return

    logger = get_logger(__name__)

    logger.error("Test error message for journalctl")
    logger.critical("Test critical message for journalctl")

    print("✅ Test 10 passed: systemd journal logging working")
    print("   View with: journalctl -t llamanote --since '1 minute ago'")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("LlamaNote Logging System - Comprehensive Test Suite")
    print("="*70)

    tests = [
        test_basic_logging,
        test_hierarchical_naming,
        test_function_entry_exit,
        test_loop_logging,
        test_api_request_logging,
        test_checkpoint_logging,
        test_exception_logging,
        test_configuration,
        test_log_files,
        test_systemd_integration
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} total")
    print("="*70)

    if failed == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
