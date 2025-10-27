#!/usr/bin/env python3
"""
Quick Start Script for LlamaNote Enhanced
Helps users get started with the system
"""

import os
import sys
import subprocess
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    required = [
        'torch',
        'transformers',
        'PyPDF2',
        'soundfile',
        'librosa'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing


def install_requirements():
    """Install requirements"""
    print("Installing requirements...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])


def create_directories():
    """Create necessary directories"""
    dirs = [
        Path.home() / ".config" / "llamanote",
        Path("output"),
        Path("logs"),
        Path("cache"),
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")


def download_sample_model():
    """Download a sample TTS model"""
    print("\nWould you like to download a sample TTS model? (recommended)")
    print("This will download microsoft/speecht5_tts (~500MB)")
    
    if input("Download? (y/n): ").lower() == 'y':
        try:
            from huggingface_hub import snapshot_download
            print("Downloading model...")
            snapshot_download(
                repo_id="microsoft/speecht5_tts",
                cache_dir=Path.home() / ".cache" / "huggingface" / "hub"
            )
            print("✓ Model downloaded successfully")
        except Exception as e:
            print(f"⚠ Could not download model: {e}")
            print("You can download it later through the menu")


def show_quick_tutorial():
    """Show quick tutorial"""
    print("\n" + "="*60)
    print("QUICK START TUTORIAL")
    print("="*60)
    
    print("\n1. LAUNCH INTERACTIVE MENU:")
    print("   python llamanote.py")
    
    print("\n2. PROCESS A PDF:")
    print("   python llamanote.py document.pdf")
    
    print("\n3. PROCESS AND GENERATE AUDIO:")
    print("   python llamanote.py document.pdf --generate-audio")
    
    print("\n4. BATCH PROCESSING:")
    print("   python llamanote.py *.pdf --generate-audio")
    
    print("\n" + "="*60)
    print("MENU NAVIGATION TIPS:")
    print("="*60)
    
    print("• Use number keys to select options")
    print("• Press 'b' to go back")
    print("• Press 'q' to quit")
    print("• Save configurations for repeated use")
    print("• Search models by language or popularity")
    
    print("\n" + "="*60)


def main():
    """Main setup function"""
    print("="*60)
    print("LLAMANOTE ENHANCED - SETUP")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("⚠ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Check requirements
    missing = check_requirements()
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        if input("Install requirements? (y/n): ").lower() == 'y':
            install_requirements()
        else:
            print("Please install requirements manually:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("✓ All required packages installed")
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Offer to download sample model
    download_sample_model()
    
    # Show tutorial
    show_quick_tutorial()
    
    print("\n✅ Setup complete!")
    print("\nLaunch the application with:")
    print("  python llamanote.py")
    
    if input("\nLaunch now? (y/n): ").lower() == 'y':
        import llamanote
        llamanote.main()


if __name__ == "__main__":
    main()
