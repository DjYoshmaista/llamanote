# LlamaNote Enhanced

<div align="center">

**🎙️ Advanced Document Processing Pipeline with AI-Powered Audio Generation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Transform documents into formatted transcripts and natural-sounding podcasts using local or cloud AI models*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 📖 Overview

LlamaNote Enhanced is a production-ready, modular document processing system that transforms PDFs, text files, and markdown documents into professionally formatted content with optional AI-generated audio narration. It combines advanced language models with sophisticated text processing to create podcast-style audio content, technical documentation, or narrative storytelling.

### What Makes It Special?

- **🎯 Flexible AI Backend**: Support for local models (HuggingFace, GGUF) and cloud APIs (OpenAI, Anthropic, Google)
- **🎙️ High-Quality Audio**: Generate natural-sounding audio using local TTS models or premium cloud voices
- **💾 Smart Checkpointing**: Resume processing from any stage - never lose progress
- **🖥️ Interactive Menu**: User-friendly terminal interface for configuration and execution
- **⚡ Memory Optimized**: Intelligent VRAM/RAM management with quantization and layer splitting
- **📊 Pipeline Stages**: Modular processing with extract, preprocess, chunk, process, filter, format, save, and audio stages

---

## ✨ Features

### 🧠 AI Model Support

#### Text Processing Models
- **Local HuggingFace**: Run models locally with full privacy (Qwen, Llama, Gemma, Phi, etc.)
- **Local GGUF**: Ultra-efficient quantized models via llama-cpp-python
- **Cloud APIs**:
  - OpenAI (GPT-3.5, GPT-4, GPT-4o)
  - Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus)
  - Google (Gemini Pro, Gemini Flash)

#### Audio Generation Models
- **Local TTS**: Privacy-first audio with Microsoft VibeVoice, SpeechT5, and others
- **OpenAI TTS**: Premium voices (Alloy, Echo, Fable, Onyx, Nova, Shimmer) with tts-1 or tts-1-hd

### 🎨 Processing Modes

| Mode | Best For | Features |
|------|----------|----------|
| **Podcast** | Audio content, storytelling | Speaker markers, emotional cues, audio-optimized cleaning |
| **Technical** | Documentation, code | Preserves technical notation, generates TOC, formats code blocks |
| **Narrative** | Stories, books | Character dialogue, scene breaks, pacing markers |
| **Summary** | Quick overviews | Concise extraction of key points |
| **Custom** | Specialized needs | User-defined prompts and formatting |

### 💾 Advanced Checkpointing System

**Never lose progress again!** The checkpoint system saves pipeline state after each stage:

- ✅ **Auto-Resume**: Automatically detects and resumes from last successful stage
- ✅ **Stage-Aware**: Intelligent compatibility checking - rerun only affected stages
- ✅ **Hash-Based Organization**: Efficient storage organized by input file and configuration
- ✅ **Interactive Management**: Browse, view, delete, and manage checkpoints via menu
- ✅ **Single-File Format**: Each checkpoint is a self-contained `.ckpt` file with metadata

**Example**: If processing fails at the "process" stage (e.g., model loading error), fix the issue and rerun - the system will skip extract, preprocess, and chunk stages, resuming directly from process!

### 🎙️ Audio Generation

Transform text into natural-sounding audio:

- **Multiple TTS Backends**: Local models or cloud APIs
- **Voice Customization**: Speed, pitch, volume normalization
- **Smart Chunking**: Handles long documents by splitting intelligently
- **Audio Post-Processing**: Built-in effects and normalization
- **Format Support**: WAV, MP3, FLAC, OGG output formats

### 🖥️ Interactive Menu System

User-friendly terminal interface with:

1. **File Browser**: Navigate directories by file type (PDF, Text, Markdown, Transcript, Checkpoint)
2. **Auto-Stage Population**: Automatically select appropriate pipeline stages based on input type
3. **Model Management**: Browse, download, and configure AI models
4. **Checkpoint Browser**: View checkpoint metadata, resume from any point
5. **Configuration Presets**: Save and load complete pipeline configurations
6. **Real-time Progress**: Live updates during processing

### 📊 Pipeline Architecture

```
┌─────────────┐
│ Input File  │ (PDF, TXT, MD, Checkpoint)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Extract    │ Parse document, extract text
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Preprocess  │ Clean text, normalize formatting
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunk     │ Split into manageable pieces
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Process    │ AI model processes each chunk
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Filter    │ Remove artifacts, merge chunks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Format    │ Apply styling (Markdown, emotions)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Save     │ Write formatted text to disk
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Audio     │ Generate audio narration (optional)
└─────────────┘
```

Each stage:
- ✅ Can be skipped or customized
- ✅ Saves a checkpoint for resume capability
- ✅ Has comprehensive error handling and logging
- ✅ Provides detailed statistics and timing

---

## 🚀 Installation

### Prerequisites

- **Python 3.8 or higher**
- **Operating System**: Linux, macOS, or Windows (WSL recommended)
- **GPU (Optional)**: NVIDIA GPU with CUDA for faster local model inference

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/llamanote-enhanced.git
cd llamanote-enhanced
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n llamanote python=3.10
conda activate llamanote
```

### Step 3: Install Dependencies

```bash
# Basic installation
pip install torch transformers accelerate bitsandbytes
pip install anthropic openai google-generativeai
pip install pypdf2 pymupdf  # PDF processing
pip install soundfile librosa  # Audio processing
pip install python-dotenv  # Environment variables

# For CUDA support (optional, for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For GGUF model support (optional)
pip install llama-cpp-python

# For local audio generation (optional)
pip install datasets  # Required by some TTS models
```

### Step 4: Set Up API Keys (Optional)

Create a `.env` file in the project root:

```bash
# OpenAI API Key
OPENAI_API_KEY=sk-...

# Anthropic API Key
ANTHROPIC_API_KEY=sk-ant-...

# Google API Key
GOOGLE_API_KEY=...

# Logging level
LOG_LEVEL=INFO
```

Or configure via the interactive menu: **Main Menu → 3. Cloud API Keys**

### Step 5: Verify Installation

```bash
python main.py --help
```

---

## 🎯 Quick Start

### Interactive Menu Mode (Recommended)

Launch the interactive menu system:

```bash
python main.py
```

**Menu Walkthrough:**

1. **Select Input & Stages** (Option 1)
   - Choose file type: PDF, Text, Markdown, Transcript, or Checkpoint
   - Browse and select your input file
   - Stages auto-populate based on file type

2. **Model Settings** (Option 2)
   - Select text processing model (local or cloud)
   - Choose audio generation model (if needed)
   - Configure memory profile for local models

3. **Configure Output** (Option 4)
   - Set output directory
   - Choose output format (Markdown, Text, JSON, HTML)

4. **Run Pipeline** (Option R)
   - Review configuration
   - Confirm and execute

### Command-Line Mode

Process a single document:

```bash
# Basic usage with defaults (Podcast mode, auto-detect model)
python main.py document.pdf

# Specify model and mode
python main.py document.pdf --model qwen3-4b --mode podcast

# Generate audio output
python main.py document.pdf --audio --audio-provider local_audio

# Use cloud API
python main.py document.pdf --provider anthropic --model claude-3-haiku-20240307
```

### Quick Examples

```bash
# 1. Convert PDF to podcast transcript
python main.py research_paper.pdf --mode podcast --output podcast.md

# 2. Generate audio from existing transcript
python main.py transcript.md --audio --audio-provider openai_audio --audio-model tts-1-hd

# 3. Process with specific stages only
python main.py document.pdf --stages extract preprocess chunk process

# 4. Resume from checkpoint
python main.py  # Use menu → Load from Checkpoint

# 5. Low VRAM configuration
python main.py document.pdf --memory-profile low_vram --quantization 4bit
```

---

## 📚 Documentation

### Configuration Files

LlamaNote Enhanced uses a hierarchical configuration system:

```
~/.config/llamanote/           # User configuration directory
├── cloud_keys.json            # API keys (encrypted)
├── presets/                   # Saved pipeline configurations
│   ├── my_podcast_preset.json
│   └── technical_docs.json
├── audio_configs/             # Audio-specific presets
│   └── high_quality_voice.json
└── profiles/                  # Memory optimization profiles
    └── custom_profile.json

project_root/
├── checkpoints/               # Pipeline checkpoints
│   ├── <input_hash>/         # Organized by input file
│   │   └── <config_hash>_<stage>.ckpt
├── output/                    # Generated files
├── cache/                     # Model and temporary files
└── logs/                      # Detailed execution logs
```

### API Key Configuration

#### Method 1: Environment Variables (.env file)

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

#### Method 2: Interactive Menu

```
Main Menu → 3. Cloud API Keys
→ Select provider → Enter API key → Save
```

Keys are securely stored in `~/.config/llamanote/cloud_keys.json`

### Model Configuration

#### Local Models (HuggingFace)

```bash
# List available models
python main.py --list-models

# Use specific model
python main.py document.pdf --model meta-llama/Llama-3.2-3B-Instruct --provider local_hf
```

**Recommended Local Models:**
- `Qwen/Qwen3-4B-Instruct-2507` - Best general purpose (32K context)
- `microsoft/phi-2` - Fast, efficient (2.7B params)
- `google/gemma-3-270m` - Ultra-fast fallback

#### Local Models (GGUF)

```bash
# Use GGUF model file
python main.py document.pdf --provider local_gguf --model /path/to/model.gguf
```

#### Cloud Models

**OpenAI:**
```bash
python main.py doc.pdf --provider openai --model gpt-4o-mini
```

**Anthropic:**
```bash
python main.py doc.pdf --provider anthropic --model claude-3-haiku-20240307
```

**Google:**
```bash
python main.py doc.pdf --provider google --model gemini-1.5-flash
```

### Memory Profiles

Optimize for your hardware:

| Profile | VRAM | Quantization | GPU Layers | Best For |
|---------|------|--------------|------------|----------|
| `low_vram` | 4GB | 4-bit | Auto | Consumer GPUs (GTX 1660, RTX 3060) |
| `medium_vram` | 8GB | 8-bit | Auto | Mid-range GPUs (RTX 3070, 4060) |
| `high_vram` | 24GB+ | None | All | Professional GPUs (RTX 4090, A6000) |
| `cpu_only` | 0GB | 8-bit | 0 | No GPU available |

**Usage:**
```bash
python main.py document.pdf --memory-profile low_vram
```

### Audio Configuration

#### Local Audio Generation

```bash
python main.py document.pdf \
  --audio \
  --audio-provider local_audio \
  --audio-model microsoft/VibeVoice-1.5B \
  --sample-rate 24000 \
  --audio-format wav
```

**Local TTS Models:**
- `microsoft/VibeVoice-1.5B` - High quality, fast
- `microsoft/speecht5_tts` - Lightweight option
- `facebook/mms-tts` - Multilingual support

#### Cloud Audio Generation (OpenAI TTS)

```bash
python main.py document.pdf \
  --audio \
  --audio-provider openai_audio \
  --audio-model tts-1-hd \
  --voice alloy \
  --audio-format mp3
```

**Available Voices:** `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

**Models:**
- `tts-1` - Fast, good quality
- `tts-1-hd` - Higher quality, slightly slower

#### Audio Post-Processing

```bash
# Adjust speed and pitch
python main.py doc.pdf --audio --speed 1.1 --pitch-shift 2

# Volume normalization
python main.py doc.pdf --audio --volume-normalize

# Different output format
python main.py doc.pdf --audio --audio-format mp3
```

### Checkpoint System

#### Auto-Resume

By default, the system automatically resumes from the latest compatible checkpoint:

```python
# In PipelineConfig
checkpoint_resume_mode = "auto"  # Default
```

Modes:
- `"auto"` - Automatically resume from latest checkpoint
- `"interactive"` - Show menu to select checkpoint
- `"disabled"` - Never resume, always start fresh

#### Manual Checkpoint Management

Via Interactive Menu:

```
Main Menu → 7. Checkpoint Management
```

Options:
1. **Browse Checkpoints** - View all available checkpoints
2. **View Checkpoint Details** - Inspect metadata and settings
3. **Delete Checkpoint** - Remove specific checkpoint
4. **Delete All for File** - Remove all checkpoints for an input file
5. **Delete All Checkpoints** - Clear entire checkpoint directory
6. **Change Checkpoint Directory** - Modify storage location
7. **Cleanup Old Checkpoints** - Keep only N most recent per stage

#### Loading from Checkpoint

**Method 1: Menu System**
```
Main Menu → 1. Select Input & Stages → 3. Load from Checkpoint
```

**Method 2: File Browser**
```
Main Menu → 1. Select Input & Stages → 1. Browse by File Type → Checkpoint
```

#### Checkpoint File Structure

```
checkpoints/
└── a1b2c3d4/              # Input file hash
    ├── e5f6g7h8_extract.ckpt
    ├── e5f6g7h8_preprocess.ckpt
    ├── e5f6g7h8_chunk.ckpt
    ├── e5f6g7h8_process.ckpt
    ├── e5f6g7h8_filter.ckpt
    ├── e5f6g7h8_format.ckpt
    ├── e5f6g7h8_save.ckpt
    └── e5f6g7h8_audio.ckpt
```

Each checkpoint contains:
- Pipeline data (text, chunks, processed output)
- Metadata (timestamp, model, settings)
- Hash information (for compatibility checking)

---

## 💡 Examples

### Example 1: Create a Podcast from Research Paper

```bash
# Step 1: Convert PDF to transcript
python main.py research_paper.pdf \
  --mode podcast \
  --model qwen3-4b \
  --chunk-size 1500 \
  --output output/podcast_transcript.md

# Step 2: Generate audio
python main.py output/podcast_transcript.md \
  --audio \
  --audio-provider openai_audio \
  --audio-model tts-1-hd \
  --voice nova
```

**Result**: `output/podcast_transcript.wav` with natural-sounding narration

### Example 2: Technical Documentation Processing

```bash
python main.py technical_manual.pdf \
  --mode technical \
  --preserve-layout \
  --no-audio-clean \
  --output docs/manual.md
```

Features preserved:
- Code blocks and inline code
- Tables and lists
- Mathematical notation
- Section hierarchy

### Example 3: Resume After Failure

```bash
# Initial run (fails at 'process' stage due to OOM)
python main.py large_document.pdf --model large-model

# Fix: Switch to smaller model and lower memory
# System automatically resumes from 'chunk' checkpoint!
python main.py large_document.pdf \
  --model qwen3-4b \
  --memory-profile low_vram
```

### Example 4: Batch Processing

```bash
# Process entire directory
python main.py documents/ --recursive --mode podcast --audio
```

### Example 5: Custom Processing Pipeline

```python
from pathlib import Path
from src.core.pipeline import ProcessingPipeline
from src.core.types import PipelineConfig, ProcessingMode, AudioConfig
from src.models.backends import get_llm_backend, get_audio_backend

# Configure pipeline
config = PipelineConfig(
    mode=ProcessingMode.PODCAST,
    model_provider="anthropic",
    model_specifier="claude-3-haiku-20240307",
    chunk_size=2000,
    generate_audio=True,
    audio_provider="openai_audio",
    audio_config=AudioConfig(
        cloud_voice="nova",
        sample_rate=24000,
        speed=1.1
    )
)

# Initialize backends
api_keys = {"anthropic": "sk-ant-...", "openai": "sk-..."}
llm_backend = get_llm_backend("anthropic", "claude-3-haiku-20240307", api_keys, config.get_hyperparameters())
audio_backend = get_audio_backend("openai_audio", "tts-1-hd", api_keys, config.audio_config)

# Create and run pipeline
pipeline = ProcessingPipeline(config)
pipeline.llm_backend = llm_backend
pipeline.audio_backend = audio_backend

result = pipeline.process_file(Path("document.pdf"))

if result.success:
    print(f"✅ Success!")
    print(f"   Transcript: {result.output_file}")
    print(f"   Audio: {result.audio_file}")
    print(f"   Time: {result.processing_time:.2f}s")
else:
    print(f"❌ Failed: {result.error_message}")
```

---

## 🛠️ Advanced Topics

### Custom System Prompts

Override default prompts for specialized processing:

```bash
python main.py document.pdf \
  --system-prompt "Convert this academic paper into a conversational podcast script suitable for teenagers. Use simple language and analogies."
```

Or via configuration file:

```json
{
  "mode": "custom",
  "system_prompt": "Your custom instructions here...",
  "model": "claude-3-sonnet-20250219"
}
```

### Chunking Strategies

Different strategies for different content types:

```bash
# Word boundary (default) - fast, simple
python main.py doc.pdf --chunk-strategy word_boundary

# Sentence boundary - better coherence
python main.py doc.pdf --chunk-strategy sentence_boundary

# Paragraph boundary - best for narrative
python main.py doc.pdf --chunk-strategy paragraph_boundary

# Semantic chunking - AI-powered, context-aware (experimental)
python main.py doc.pdf --chunk-strategy semantic
```

### Hyperparameter Tuning

Fine-tune model generation:

```python
# Via menu: Model Settings → Advanced → Hyperparameters

# Or programmatically:
from src.models.hyperparameters import HyperparameterConfig

hyperparams = HyperparameterConfig(
    temperature=0.7,        # Creativity (0.0-2.0)
    top_p=0.9,             # Nucleus sampling
    top_k=50,              # Top-k sampling
    max_new_tokens=2000,   # Max output length
    repetition_penalty=1.1 # Avoid repetition
)
```

### Custom Formatters

Create your own formatting style:

```python
# src/formatting/my_formatter.py
from src.formatting.base_formatter import BaseFormatter

class MyCustomFormatter(BaseFormatter):
    def format(self, text, add_emotions=True, add_structure=True):
        # Your custom formatting logic
        return formatted_text

# Register in src/formatting/__init__.py
```

### Model Registry

Add custom models:

```python
# src/models/registry.py
MODEL_REGISTRY = {
    "my-custom-model": ModelEntry(
        name="My Custom Model",
        model_id="username/model-name-hf",
        author="username",
        supports_thinking=True,
        max_context=8192,
        optimal_chunk_size=1000
    )
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Error

**Symptoms:** CUDA out of memory, Process killed

**Solutions:**

```bash
# Use lower memory profile
python main.py doc.pdf --memory-profile low_vram

# Enable 4-bit quantization
python main.py doc.pdf --quantization 4bit

# Smaller chunk size
python main.py doc.pdf --chunk-size 500

# Use CPU only
python main.py doc.pdf --memory-profile cpu_only
```

#### 2. Model Download Fails

**Symptoms:** Connection timeout, 403 Forbidden

**Solutions:**

```bash
# For gated models (Llama, etc.), authenticate with HuggingFace
huggingface-cli login

# Check HuggingFace model access permissions
# Visit: https://huggingface.co/<model-id>

# Use alternative model
python main.py doc.pdf --model gemma-270m  # No authentication needed
```

#### 3. Audio Generation Fails

**Symptoms:** "array is too big", OpenAI TTS API errors

**Solutions:**

```bash
# For local audio OOM
python main.py doc.pdf --audio --audio-quantization 4bit --audio-cpu-offload

# For OpenAI FLAC issues (fixed in latest version)
# System automatically uses MP3 format now

# Check API key
echo $OPENAI_API_KEY

# Use different provider
python main.py doc.pdf --audio --audio-provider local_audio
```

#### 4. Checkpoint Compatibility Issues

**Symptoms:** Checkpoint not loading, "incompatible" warnings

**Reason:** Configuration changed (different model, chunk size, etc.)

**Solutions:**

```bash
# View checkpoint details via menu
# Main Menu → 7. Checkpoint Management → 2. View Details

# Delete incompatible checkpoints
# Main Menu → 7. Checkpoint Management → 3. Delete Checkpoint

# Or disable checkpointing for this run
python main.py doc.pdf --no-checkpoints
```

#### 5. Slow Processing

**Solutions:**

```bash
# Use smaller/faster model
python main.py doc.pdf --model gemma-270m

# Use cloud API (usually faster)
python main.py doc.pdf --provider anthropic --model claude-3-haiku-20240307

# Enable quantization
python main.py doc.pdf --quantization 4bit

# Larger chunks (fewer API calls)
python main.py doc.pdf --chunk-size 2000
```

### Debug Mode

Enable detailed logging:

```bash
# Method 1: Command line
python main.py doc.pdf --verbose

# Method 2: Environment variable
export LOG_LEVEL=DEBUG
python main.py doc.pdf

# Method 3: Edit .env file
LOG_LEVEL=DEBUG
```

Logs are saved to `logs/` directory with timestamps.

### Getting Help

1. **Check logs:** `logs/processing_log_<date>.log`
2. **Review checkpoint:** Menu → Checkpoint Management → View Details
3. **Check system info:**
   ```bash
   python main.py --system-info
   ```
4. **Report issues:** Include logs, configuration, and error messages

---

## 🏗️ Architecture

### Project Structure

```
llamanote-enhanced/
├── main.py                   # Entry point
├── src/
│   ├── cli.py               # Command-line interface
│   ├── menu.py              # Interactive menu system
│   ├── menu_checkpoint.py   # Checkpoint management UI
│   ├── config/              # Configuration management
│   │   ├── settings.py      # Default settings
│   │   ├── manager.py       # Config CRUD operations
│   │   ├── profiles.py      # Memory profiles
│   │   └── presets.py       # Hyperparameter presets
│   ├── core/                # Core pipeline logic
│   │   ├── pipeline.py      # Main processing pipeline
│   │   ├── stages.py        # Individual stage implementations
│   │   ├── types.py         # Data types and enums
│   │   └── errors.py        # Custom exceptions
│   ├── models/              # Model backends
│   │   ├── registry.py      # Model registry
│   │   ├── hub.py           # Model download/management
│   │   ├── hyperparameters.py
│   │   └── backends/
│   │       ├── base.py      # Abstract base classes
│   │       ├── local_hf.py  # HuggingFace models
│   │       ├── local_gguf.py # GGUF models
│   │       ├── openai_llm.py
│   │       ├── anthropic_llm.py
│   │       ├── google_llm.py
│   │       ├── local_audio.py
│   │       └── openai_audio.py
│   ├── processing/          # Text processing
│   │   ├── pdf_extractor.py
│   │   ├── text_preprocessor.py
│   │   ├── text_chunker.py
│   │   ├── response_filter.py
│   │   └── audio_processor.py
│   ├── formatting/          # Output formatting
│   │   ├── base_formatter.py
│   │   ├── podcast_formatter.py
│   │   ├── technical_formatter.py
│   │   └── narrative_formatter.py
│   ├── io/                  # Input/output operations
│   │   ├── file_handler.py
│   │   ├── batch_manager.py
│   │   └── checkpoints.py   # Checkpoint system
│   └── utils/               # Utilities
│       ├── logger.py        # Logging system
│       ├── file_browser.py  # Interactive file browser
│       ├── validators.py
│       └── decorators.py
├── checkpoints/             # Pipeline checkpoints
├── output/                  # Generated files
├── logs/                    # Execution logs
└── cache/                   # Model cache and temp files
```

### Data Flow

```
Input File (PDF/TXT/MD/Checkpoint)
    ↓
┌─────────────────────────────────────┐
│ Extract Stage                       │
│ - PDFProcessor (PyMuPDF)           │
│ - Text file reader                  │
│ Output: ExtractionResult           │
└─────────────────────────────────────┘
    ↓ Checkpoint Saved
┌─────────────────────────────────────┐
│ Preprocess Stage                    │
│ - TextPreprocessor                  │
│ - Clean for audio (optional)        │
│ Output: Cleaned text string         │
└─────────────────────────────────────┘
    ↓ Checkpoint Saved
┌─────────────────────────────────────┐
│ Chunk Stage                         │
│ - TextChunker                       │
│ - Strategy: word/sentence/paragraph │
│ Output: List of text chunks         │
└─────────────────────────────────────┘
    ↓ Checkpoint Saved
┌─────────────────────────────────────┐
│ Process Stage                       │
│ - LLM Backend (local or cloud)      │
│ - Batch processing of chunks        │
│ Output: Processed chunks            │
└─────────────────────────────────────┘
    ↓ Checkpoint Saved
┌─────────────────────────────────────┐
│ Filter Stage                        │
│ - ResponseFilter                    │
│ - Remove thinking tokens            │
│ Output: Merged, filtered text       │
└─────────────────────────────────────┘
    ↓ Checkpoint Saved
┌─────────────────────────────────────┐
│ Format Stage                        │
│ - Formatter (Podcast/Technical/etc) │
│ - Add emotions, structure           │
│ Output: Formatted markdown          │
└─────────────────────────────────────┘
    ↓ Checkpoint Saved
┌─────────────────────────────────────┐
│ Save Stage                          │
│ - FileHandler                       │
│ - Write to output directory         │
│ Output: Saved file path             │
└─────────────────────────────────────┘
    ↓ Checkpoint Saved
┌─────────────────────────────────────┐
│ Audio Stage (Optional)              │
│ - Audio Backend (local or cloud)    │
│ - AudioPostProcessor                │
│ Output: Audio file (WAV/MP3)        │
└─────────────────────────────────────┘
    ↓ Checkpoint Saved
Final Output:
  - Formatted text file
  - Audio file (if generated)
  - Processing report
  - Statistics
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Model Support**: Add more model backends (Mistral, Cohere, etc.)
- **Audio Providers**: Integrate additional TTS services (ElevenLabs, Azure TTS)
- **Semantic Chunking**: Implement better context-aware chunking
- **Formatting**: Create new formatting styles
- **UI**: Improve interactive menu (TUI with rich/textual)
- **Multi-language**: Add support for non-English documents
- **Testing**: Expand test coverage

### Development Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/yourusername/llamanote-enhanced.git
cd llamanote-enhanced
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Technologies
- **[Transformers](https://github.com/huggingface/transformers)** by Hugging Face - Model loading and inference
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** - GGUF model support
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - PDF text extraction
- **[Librosa](https://librosa.org/)** - Audio processing

### APIs
- **[OpenAI](https://openai.com/)** - GPT models and TTS
- **[Anthropic](https://anthropic.com/)** - Claude models
- **[Google AI](https://ai.google/)** - Gemini models

### Model Creators
- Qwen Team (Alibaba Cloud)
- Google (Gemma models)
- Meta (Llama models)
- Microsoft (Phi, SpeechT5, VibeVoice)

---

## 📞 Support

### Getting Help

- **Documentation**: This README and inline code comments
- **Logs**: Check `logs/` directory for detailed execution logs
- **Issues**: [GitHub Issues](https://github.com/yourusername/llamanote-enhanced/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llamanote-enhanced/discussions)

### Useful Commands

```bash
# System information
python main.py --system-info

# List available models
python main.py --list-models

# Test configuration
python main.py --test-config

# Check API keys
python main.py --check-keys

# View help
python main.py --help
```

---

<div align="center">

**Made with ❤️ by the LlamaNote Enhanced Team**

⭐ Star this repo if you find it useful!

[Report Bug](https://github.com/yourusername/llamanote-enhanced/issues) • [Request Feature](https://github.com/yourusername/llamanote-enhanced/issues) • [Documentation](https://github.com/yourusername/llamanote-enhanced/wiki)

</div>
