## LlamaNote Enhanced - Advanced PDF to Formatted Text Processor

A highly modular, production-ready system for processing PDF documents into formatted text using advanced language models with thinking model support, emotional markers, and multiple output formats.

## ✨ Key Features

### 🧠 Advanced Model Support
- **Thinking Model Filtering**: Automatically detects and removes thinking tokens from models like Qwen3-4B
- **Multiple Model Support**: Compatible with Qwen, Gemma, Llama, and other models
- **Quantization Support**: 4-bit and 8-bit quantization for reduced VRAM usage
- **Auto Device Mapping**: Intelligent GPU/CPU memory management

### 📝 Text Processing
- **Multiple Chunking Strategies**: Word, sentence, paragraph, semantic, sliding window
- **Smart Preprocessing**: Audio-specific cleaning, LaTeX conversion, table handling
- **Response Filtering**: Removes thinking artifacts, acknowledgments, and unwanted patterns

### 🎨 Advanced Formatting
- **Emotional Markers**: Detect and mark emotions in text (excited, thoughtful, serious, etc.)
- **Multiple Styles**: Podcast, technical, narrative formatting
- **Speaker Detection**: Automatic dialogue formatting for podcasts
- **Markdown Enhancements**: Custom emphasis levels, structure markers, TOC generation

### 🚀 Production Features
- **Extensive Logging**: Detailed logging with context tracking and performance monitoring
- **Error Handling**: Automatic retries, fallback mechanisms, comprehensive error reporting
- **Checkpointing**: Save and resume processing at any stage
- **Batch Processing**: Handle multiple files with progress tracking
- **Memory Management**: Automatic VRAM/RAM optimization and monitoring

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llamanote-enhanced.git
cd llamanote-enhanced

# Install dependencies
pip install -r requirements.txt

# For CUDA support (if using GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

### Basic Usage

```bash
# Process a single PDF with defaults (podcast mode)
python llamanote_enhanced.py document.pdf

# Process with specific model and chunk size
python llamanote_enhanced.py document.pdf --model qwen3-4b --chunk-size 1500

# Process in technical documentation mode
python llamanote_enhanced.py document.pdf -m technical

# Process entire directory recursively
python llamanote_enhanced.py documents/ -r -m podcast
```

### Advanced Usage

```bash
# Low VRAM configuration (4GB GPU)
python llamanote_enhanced.py document.pdf --memory-profile low_vram

# Custom system prompt
python llamanote_enhanced.py document.pdf --system-prompt "Convert this text for a children's audiobook"

# Multiple output formats
python llamanote_enhanced.py document.pdf -f html -m narrative

# Preserve thinking tokens (for debugging)
python llamanote_enhanced.py document.pdf --no-thinking

# Use configuration file
python llamanote_enhanced.py document.pdf --config my_config.json
```

## 🎯 Processing Modes

### Podcast Mode
- Cleans text for audio narration
- Adds speaker markers
- Detects emotions and pacing
- Removes technical elements (URLs, math, tables)

### Technical Mode
- Preserves code blocks and technical notation
- Generates table of contents
- Formats inline code and notes
- Maintains document structure

### Narrative Mode
- Emphasis on storytelling flow
- Character dialogue formatting
- Scene and chapter breaks
- Emotional tone markers

## ⚙️ Configuration

### Memory Profiles

| Profile | VRAM | Quantization | Use Case |
|---------|------|--------------|----------|
| `low_vram` | 4GB | 4-bit | Consumer GPUs |
| `medium_vram` | 8GB | 8-bit | Mid-range GPUs |
| `high_vram` | 24GB | None | Professional GPUs |
| `cpu_only` | 0GB | 8-bit | No GPU available |

### Available Models

| Model | Thinking Support | Context | Best For |
|-------|-----------------|---------|----------|
| `qwen3-4b` | ✅ Yes | 32K | General purpose |
| `gemma-270m` | ❌ No | 8K | Fast processing |
| `llama-3.2-1b` | ❌ No | 8K | Balanced |

## 📊 Command Line Options

```
Usage: llamanote_enhanced.py [OPTIONS] INPUT [INPUT ...]

Positional Arguments:
  INPUT                 Input PDF file(s) or directory

Optional Arguments:
  -o, --output PATH     Output file path
  -f, --format FORMAT   Output format (markdown/text/json/html)
  -m, --mode MODE       Processing mode (podcast/technical/narrative)
  -r, --recursive       Process directories recursively

Model Options:
  --model MODEL         Model to use
  --memory-profile      Memory optimization profile

Processing Options:
  --chunk-size SIZE     Target chunk size (100-5000)
  --chunk-strategy      Chunking strategy
  --no-thinking         Keep thinking tokens
  --preserve-layout     Preserve PDF layout
  --system-prompt       Custom system prompt

Formatting Options:
  --markdown-style      Markdown formatting style
  --no-emotions         Disable emotional markers
  --no-audio-clean      Disable audio cleaning

Performance Options:
  --no-checkpoints      Disable checkpointing
  --max-retries N       Max retries for failures
  -v, --verbose         Verbose output
```

## 📁 Output Examples

### Podcast Format
```markdown
## Processed Content

**[Speaker Host]:**

🎉 Welcome to today's fascinating discussion about artificial intelligence!

**[Speaker Guest]:**

🤔 I've been thinking about how AI is transforming our daily lives...

... [pause for emphasis] ... 

It's truly remarkable how far we've come.
```

### Technical Format
```markdown
## Table of Contents

1. [Introduction](#introduction)
2. [Implementation](#implementation)
3. [Results](#results)

## Introduction

This document describes the implementation of `neural_network()` architecture.

📝 **Note:** All code examples use Python 3.8+

⚠️ **Warning:** Ensure CUDA drivers are updated before running.
```

## 🔧 Customization

### Custom Configuration File

Create a `config.json`:

```json
{
  "mode": "podcast",
  "model": "qwen3-4b",
  "memory_profile": "medium_vram",
  "chunk_size": 1500,
  "chunk_strategy": "sentence_boundary",
  "markdown_style": "podcast",
  "remove_thinking": true,
  "add_emotions": true,
  "output_format": "markdown"
}
```

### Using as a Module

```python
from processing_pipeline import ProcessingPipeline, PipelineConfig, ProcessingMode
from pathlib import Path

# Configure pipeline
config = PipelineConfig(
    mode=ProcessingMode.PODCAST,
    model_name="qwen3-4b",
    chunk_size=1500,
    remove_thinking=True,
    add_emotions=True
)

# Create and run pipeline
pipeline = ProcessingPipeline(config)
result = pipeline.process_file(Path("document.pdf"))

if result.success:
    print(f"Success! Output: {result.output_file}")
    print(f"Processing time: {result.processing_time:.2f}s")
else:
    print(f"Failed: {result.error_message}")
```

## 📈 Performance Tips

1. **VRAM Management**
   - Use appropriate memory profile for your GPU
   - Enable quantization for large models
   - Process in smaller chunks if running out of memory

2. **Speed Optimization**
   - Use smaller models for draft processing
   - Enable checkpointing for long documents
   - Adjust chunk size based on model context window

3. **Quality Optimization**
   - Use sentence or paragraph boundaries for better coherence
   - Enable emotional markers for engaging content
   - Custom prompts for specific use cases

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Use lower memory profile
python llamanote_enhanced.py doc.pdf --memory-profile low_vram

# Or use smaller chunks
python llamanote_enhanced.py doc.pdf --chunk-size 500
```

**Thinking Tokens in Output**
```bash
# Make sure thinking removal is enabled (default)
python llamanote_enhanced.py doc.pdf

# Check model configuration for new thinking patterns
# Add custom patterns in config.py THINKING_PATTERNS
```

**Slow Processing**
```bash
# Use quantization
python llamanote_enhanced.py doc.pdf --memory-profile low_vram

# Or use a smaller model
python llamanote_enhanced.py doc.pdf --model gemma-270m
```

## 📊 Processing Pipeline

```
PDF Input → Text Extraction → Preprocessing → Chunking → 
LLM Processing → Response Filtering → Formatting → Output
```

Each stage can be checkpointed and resumed, with detailed logging and error handling.

## 🤝 Contributing

Contributions are welcome! Key areas for improvement:

- Additional model support
- Enhanced semantic chunking
- More formatting styles
- Multi-language support
- API endpoint implementation

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Transformers library by Hugging Face
- PyPDF2 and PyMuPDF communities
- Model creators (Qwen, Gemma, Llama teams)

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the detailed logs in `logs/` directory
- Review checkpoint files in `cache/checkpoints/` for debugging

---

**Note**: This is an enhanced version of the original LlamaNote script with extensive improvements in modularity, error handling, logging, and feature set. The system is designed for production use with comprehensive testing and optimization capabilities.


--- 
# *** OLD 
---

## Description
 - Open-source alternative to NotebookLM, complete with RAG database inferencing and audio file generation (podcast format)
 - Complete database interfacing with either local or remote/cloud LLMs
 - Transcript generation, alteration, and specialization
 - Audio file generation via local and/or cloud/remote LLM's (including from major LLM providers such as Claude, OpenAI, Google, and more)
 - Customizable prompts, generations, and more

## Getting Started
 - Create/start a new python virtual environment using your tool of choice (conda, uv, venv, etc)
 - Run 'pip install -r requirements.txt'
 - Edit any and all necessary files (there will be a .env creator which walks you through setting environment variables) and configurations
 - Indicate a folder and/or file(s) used for inferencing/transcript generation
 - Allow the program(s) to run and read the file(s), generate transcript(s), then finally generate audio files
 - Enjoy!
