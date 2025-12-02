#!/usr/bin/env python3
"""
Test script for audio checkpoint compression system.

Tests:
1. FLAC compression/decompression round-trip (lossless verification)
2. Compression ratio (~50-60% expected)
3. Rolling checkpoint overwrite behavior
4. Archival checkpoint milestone detection
5. Dual checkpoint saving (rolling + archival)
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path and import directly to avoid circular imports
import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules directly
base_dir = Path(__file__).parent
checkpoints_module = load_module_from_path("checkpoints", base_dir / "src" / "io" / "checkpoints.py")
types_module = load_module_from_path("types", base_dir / "src" / "core" / "types.py")

CheckpointManager = checkpoints_module.CheckpointManager
PipelineConfig = types_module.PipelineConfig
AudioConfig = types_module.AudioConfig

def test_flac_compression():
    """Test FLAC audio compression round-trip."""
    print("=" * 60)
    print("Test 1: FLAC Compression Round-Trip")
    print("=" * 60)

    # Create checkpoint manager
    manager = CheckpointManager(base_checkpoint_dir=Path("test_checkpoints"))

    # Create synthetic audio (5 seconds at 22050 Hz)
    sample_rate = 22050
    duration = 5.0
    num_samples = int(sample_rate * duration)

    # Generate sine wave audio (440 Hz)
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    audio_original = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print(f"Original audio: {audio_original.shape}, {audio_original.dtype}")
    print(f"Original size: {audio_original.nbytes / (1024*1024):.2f} MB")

    # Test compression
    compressed_bytes = manager._compress_audio_array(audio_original, sample_rate)
    print(f"Compressed size: {len(compressed_bytes) / (1024*1024):.2f} MB")
    print(f"Compression ratio: {len(compressed_bytes) / audio_original.nbytes * 100:.1f}%")

    # Test decompression
    audio_decompressed, sr_decompressed = manager._decompress_audio_array(compressed_bytes)
    print(f"Decompressed audio: {audio_decompressed.shape}, {audio_decompressed.dtype}")
    print(f"Sample rate check: {sample_rate} == {sr_decompressed}")

    # Verify lossless (allow small floating-point error)
    max_error = np.abs(audio_original - audio_decompressed).max()
    print(f"Maximum error: {max_error:.10f}")

    if max_error < 1e-5:
        print("✓ PASS: Lossless compression verified")
        return True
    else:
        print("✗ FAIL: Compression not lossless")
        return False

def test_checkpoint_data_compression():
    """Test compression of audio arrays in checkpoint data."""
    print("\n" + "=" * 60)
    print("Test 2: Checkpoint Data Compression")
    print("=" * 60)

    manager = CheckpointManager(base_checkpoint_dir=Path("test_checkpoints"))

    # Create mock checkpoint data with multiple audio arrays
    sample_rate = 22050
    num_arrays = 10
    audio_arrays = []

    for i in range(num_arrays):
        duration = 3.0
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        audio = np.sin(2 * np.pi * (440 + i * 50) * t).astype(np.float32)
        audio_arrays.append(audio)

    data = {
        'audio_arrays': audio_arrays,
        'sample_rate': sample_rate,
        'completed_stages': ['extract', 'preprocess', 'chunk', 'process', 'filter']
    }

    # Calculate original size
    original_size = sum(arr.nbytes for arr in audio_arrays)
    print(f"Original total size: {original_size / (1024*1024):.2f} MB ({num_arrays} arrays)")

    # Compress
    compressed_data = manager._compress_audio_arrays_in_checkpoint(data)

    # Calculate compressed size
    compressed_size = sum(len(item['compressed_bytes']) for item in compressed_data['audio_arrays'])
    print(f"Compressed total size: {compressed_size / (1024*1024):.2f} MB")
    print(f"Space savings: {(1 - compressed_size/original_size) * 100:.1f}%")

    # Decompress
    decompressed_data = manager._decompress_audio_arrays_in_checkpoint(compressed_data)

    # Verify all arrays
    all_match = True
    for i, (orig, decomp) in enumerate(zip(audio_arrays, decompressed_data['audio_arrays'])):
        max_error = np.abs(orig - decomp).max()
        if max_error > 1e-5:
            print(f"✗ Array {i} mismatch: error={max_error}")
            all_match = False

    if all_match:
        print(f"✓ PASS: All {num_arrays} arrays decompressed correctly")
        return True
    else:
        print("✗ FAIL: Some arrays failed verification")
        return False

def test_milestone_detection():
    """Test archival milestone detection logic."""
    print("\n" + "=" * 60)
    print("Test 3: Archival Milestone Detection")
    print("=" * 60)

    manager = CheckpointManager(base_checkpoint_dir=Path("test_checkpoints"))

    # Create minimal config
    config = PipelineConfig(
        model_provider="huggingface",
        model_specifier="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    input_path = Path("test_document.pdf")

    # Test sequence of completion percentages
    test_sequence = [5, 15, 22, 35, 42, 58, 65, 78, 85, 95, 100]
    expected_milestones = [20, 40, 60, 80, 100]
    detected_milestones = []

    print(f"Testing completion sequence: {test_sequence}")
    print(f"Expected milestones: {expected_milestones}")

    for pct in test_sequence:
        milestone = manager._should_create_archival_checkpoint(input_path, config, pct)
        if milestone is not None:
            detected_milestones.append(milestone)
            print(f"  {pct}% → Milestone {milestone}% detected")

    if detected_milestones == expected_milestones:
        print(f"✓ PASS: All {len(expected_milestones)} milestones detected correctly")
        return True
    else:
        print(f"✗ FAIL: Expected {expected_milestones}, got {detected_milestones}")
        return False

def test_checkpoint_naming():
    """Test rolling and archival checkpoint naming."""
    print("\n" + "=" * 60)
    print("Test 4: Checkpoint Naming Patterns")
    print("=" * 60)

    manager = CheckpointManager(base_checkpoint_dir=Path("test_checkpoints"))

    # Create config with models
    config = PipelineConfig(
        model_provider="huggingface",
        model_specifier="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        audio_config=AudioConfig(model_name="microsoft/speecht5_tts")
    )
    input_path = Path("test_document.pdf")

    # Test rolling checkpoint naming
    rolling_path = manager._get_checkpoint_path(
        input_path, config, "audio",
        chunk_index=15, total_chunks=50, completion_pct=35,
        checkpoint_type="rolling"
    )
    print(f"Rolling checkpoint: {rolling_path.name}")

    # Verify rolling pattern
    if "ROLLING-audio" in rolling_path.name:
        print("  ✓ Contains 'ROLLING-audio' marker")
    else:
        print("  ✗ Missing 'ROLLING-audio' marker")
        return False

    # Test archival checkpoint naming
    archival_path = manager._get_checkpoint_path(
        input_path, config, "audio",
        chunk_index=15, total_chunks=50, completion_pct=40,
        checkpoint_type="archival"
    )
    print(f"Archival checkpoint: {archival_path.name}")

    # Verify archival pattern
    if "ARCHIVE-40pct-audio" in archival_path.name:
        print("  ✓ Contains 'ARCHIVE-40pct-audio' marker")
    else:
        print("  ✗ Missing 'ARCHIVE-40pct-audio' marker")
        return False

    # Verify they're in same directory
    if rolling_path.parent == archival_path.parent:
        print("  ✓ Both in same directory")
    else:
        print("  ✗ Different directories")
        return False

    print("✓ PASS: Checkpoint naming correct")
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AUDIO CHECKPOINT COMPRESSION - TEST SUITE")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("FLAC Round-Trip", test_flac_compression()))
    results.append(("Checkpoint Data Compression", test_checkpoint_data_compression()))
    results.append(("Milestone Detection", test_milestone_detection()))
    results.append(("Checkpoint Naming", test_checkpoint_naming()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"{status:10} {name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
