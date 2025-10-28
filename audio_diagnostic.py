# Audio Generation Diagnostic tool

import sys
import os
from pathlib import Path
import numpy as np

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_audio_backend():
    # Comprehensive test of audio backend functionality
    print("="*60)
    print("AUDIO GENERATION DIAGNOSTICS TOOL")
    print("="*60)

    # Step 1: Check imports
    print("\n[1/7] Checking required imports...")
    try:
        import torch
        print(f"  ✓ Pytorch {torch.__version__}")
        print(f"    CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        from transformers import pipeline
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers import failed: {e}")
        return False

    try:
        import soundfile as sf
        print(f"  ✓ soundfile {sf.__version__}")
    except ImportError as e:
        print(f"  ✗ soundfile import failed: {e}")
        return False

    try:
        import librosa
        print(f"  ✓ librosa {librosa.__version__}")
    except ImportError as e:
        print(f"  ✗ librosa import failed: {e}")
        return False

    # Step 2: Test basic TTS model loading
    print("\n[2/7] Testing TTS model loading...")
    print("  Model microsoft/speecht5_tts (common)")

    try:
        device = 0 if torch.cuda.is_available() else -1
        print(f"  Device: {'GPU' if device == 0 else 'CPU'}")

        tts_pipeline = pipeline("text-to-speech", model="microsoft/speecht5_tts", device=device)
        print("  ✓ Model loaded successfully")
        print(f"     Pipeline type: {type(tts_pipeline).__name__}")

    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Test generation
    print("\n[3/7] Testing audio generation...")
    test_text = "This is a test"

    try:
        print(f"  Generating audio for: '{test_text}'")
        output = tts_pipeline(test_text)
        print(f"  ✓ Generation completed")
        print(f"    Output type: {type(output)}")

    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Analyze output structure
    print("\n[4/7] Analyzing output structure...")

    audio_array = None
    sample_rate = None

    if isinstance(output, dict):
        print(f"  Output is dict with keys: {list(output.keys())}")

        # Check for audio data
        audio_keys = ['audio', 'waveform', 'generated_audio', 'speech']
        for key in audio_keys:
            if key in output:
                audio_array = output[key]
                print(f"  ✓ Found audio data under key: '{key}'")
                break
        
        if audio_array is None:
            print("  ⚠ No standard audio key found")
            print("  Inspecting all keys:")
            for k, v in output.items():
                print(f"    - {k}: type={type(v)}", end="")
                if isinstance(v, np.ndarray):
                    print(f", shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f", length={len(v)}")
                else:
                    print()
        
        # Check for sample rate
        sr_keys = ['sampling_rate', 'sample_rate', 'sr']
        for key in sr_keys:
            if key in output:
                sample_rate = output[key]
                print(f"  ✓ Found sample rate under key: '{key}' = {sample_rate}Hz")
                break
        
    elif isinstance(output, (np.ndarray, list)):
        print(f"  Output is direct array/list")
        audio_array = output
        
    elif isinstance(output, tuple):
        print(f"  Output is tuple with {len(output)} elements")
        if len(output) >= 1:
            audio_array = output[0]
            print(f"    Element 0: {type(audio_array)}")
        if len(output) >= 2:
            sample_rate = output[1] if isinstance(output[1], (int, float)) else None
            print(f"    Element 1: {type(output[1])} = {output[1]}")
    
    else:
        print(f"  ✗ Unexpected output type: {type(output)}")
        return False
    
    # Step 5: Validate audio array
    print("\n[5/7] Validating audio array...")
    
    if audio_array is None:
        print("  ✗ Audio array is None")
        return False
    
    # Convert to numpy if needed
    if not isinstance(audio_array, np.ndarray):
        try:
            audio_array = np.array(audio_array)
            print(f"  Converted to numpy array")
        except Exception as e:
            print(f"  ✗ Conversion failed: {e}")
            return False
    
    print(f"  ✓ Audio array valid")
    print(f"    Shape: {audio_array.shape}")
    print(f"    Dtype: {audio_array.dtype}")
    print(f"    Size: {audio_array.size} samples")
    print(f"    Range: [{audio_array.min():.6f}, {audio_array.max():.6f}]")
    
    if audio_array.ndim != 1:
        print(f"  ⚠ Warning: Audio is not 1D (shape: {audio_array.shape})")
        audio_array = audio_array.squeeze()
        print(f"    After squeeze: {audio_array.shape}")
    
    # Step 6: Check sample rate
    print("\n[6/7] Checking sample rate...")
    
    if sample_rate is None:
        print("  ⚠ Sample rate not found in output")
        
        # Try to get from model config
        if hasattr(tts_pipeline, 'model') and hasattr(tts_pipeline.model, 'config'):
            config = tts_pipeline.model.config
            for attr in ['sampling_rate', 'sample_rate', 'sr']:
                if hasattr(config, attr):
                    sample_rate = getattr(config, attr)
                    print(f"  ✓ Got sample rate from model.config.{attr}: {sample_rate}Hz")
                    break
        
        if sample_rate is None:
            sample_rate = 16000  # Common default
            print(f"  ⚠ Using default sample rate: {sample_rate}Hz")
    else:
        print(f"  ✓ Sample rate: {sample_rate}Hz")
    
    # Step 7: Test saving
    print("\n[7/7] Testing file save...")
    
    test_output = Path("diagnostic_test_output.wav")
    try:
        # Ensure audio is in correct format for soundfile
        audio_to_save = audio_array.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio_to_save).max()
        if max_val > 1.0:
            audio_to_save = audio_to_save / max_val
            print(f"  Normalized audio (max was {max_val:.3f})")
        
        sf.write(test_output, audio_to_save, sample_rate)
        print(f"  ✓ Audio saved to: {test_output}")
        print(f"    File size: {test_output.stat().st_size} bytes")
        print(f"    Duration: {len(audio_array) / sample_rate:.2f} seconds")
        
    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print(f"\nTest audio file created: {test_output.absolute()}")
    print("You can play this file to verify audio generation is working.")
    
    return True


def test_specific_model(model_name: str):
    """Test a specific TTS model"""
    print(f"\n{'='*60}")
    print(f"TESTING SPECIFIC MODEL: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        import torch
        from transformers import pipeline
        
        device = 0 if torch.cuda.is_available() else -1
        print(f"Device: {'GPU' if device == 0 else 'CPU'}")
        
        print(f"Loading model: {model_name}")
        tts_pipeline = pipeline(
            "text-to-speech",
            model=model_name,
            device=device,
            trust_remote_code=True
        )
        print("✓ Model loaded")
        
        print("\nGenerating test audio...")
        output = tts_pipeline("Hello world")
        print(f"✓ Generation complete")
        print(f"Output type: {type(output)}")
        
        if isinstance(output, dict):
            print(f"Output keys: {list(output.keys())}")
            for k, v in output.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: numpy array, shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k}: {type(v)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Generation Diagnostic Tool")
    parser.add_argument("--model", help="Test a specific model by name/ID")
    args = parser.parse_args()
    
    if args.model:
        success = test_specific_model(args.model)
    else:
        success = test_audio_backend()
    
    sys.exit(0 if success else 1)
