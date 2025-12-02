# LlamaNote

## Description

LlamaNote is an open-source, local-first alternative to services like NotebookLM. It allows you to chat with your documents, generate detailed notes, and create high-quality audio summaries and podcasts using a variety of local and cloud-based Large Language Models (LLMs).

## Features

-   **📝 Rich Text & PDF Processing**: Extract and process text from PDFs, text files, and markdown documents.
-   **🤖 Flexible LLM Backends**: Connect to local Hugging Face models (Transformers or GGUF), or cloud providers like OpenAI, Anthropic, and Google.
-   **🎙️ High-Quality Audio Generation**: Convert your generated text into audio using local TTS models or cloud services.
-   **🎛️ Interactive Menu**: A comprehensive, easy-to-use command-line interface for controlling every aspect of the pipeline.
-   **💾 Checkpoint System**: Automatically save progress and resume long-running tasks without losing work.

### Multi-Speaker TTS

Bring your documents to life with distinct voices for each speaker. LlamaNote's advanced multi-speaker TTS system can:

-   Automatically detect speakers in your text (e.g., `[Speaker 1]`, `[Host]`).
-   Assign a unique, consistent voice to each speaker.
-   Generate voices randomly or sample from a dataset of over 7,000 real human voices.
-   Cache voices for consistency across multiple runs.

For more details, see the [Speaker Embedding Guide](./documentation/SPEAKER_EMBEDDING_GUIDE.md).

### Advanced GPU Optimization

Run powerful, multi-billion parameter models on consumer hardware with as little as 4GB of VRAM.

-   **Auto OOM Handling**: Automatically recovers from out-of-memory errors by offloading model layers to CPU RAM.
-   **Advanced Cache System**: A custom hybrid KV-cache with a sliding window that intelligently pages data between GPU and CPU, allowing for the processing of very long documents.
-   **Hardware Profiles**: Simple presets (`low_vram`, `medium_vram`, etc.) to automatically configure settings for your hardware.

For more details, see the [GPU Optimization Guide](./documentation/GPU_OPTIMIZATION_GUIDE.md).

## Documentation

- **[Architecture Guide](./documentation/ARCHITECTURE.md)** - Comprehensive codebase architecture, module structure, and code reuse opportunities
- **[Chat Template Guide](./documentation/CHAT_TEMPLATE_GUIDE.md)** - Chat template management system for batch processing
- **[Speaker Embedding Guide](./documentation/SPEAKER_EMBEDDING_GUIDE.md)** - Multi-speaker TTS system documentation
- **[GPU Optimization Guide](./documentation/GPU_OPTIMIZATION_GUIDE.md)** - Advanced GPU memory management

## Getting Started

1.  **Create a Virtual Environment**: Use your preferred tool (conda, venv, etc.).
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application**:
    ```bash
    python main.py
    ```
4.  **Follow the Menu**: The interactive menu will guide you through selecting files, choosing models, and running the processing pipeline.

## How to Use

1.  **Select Input & Stages**: Choose the documents you want to process and which parts of the pipeline to run (e.g., text generation, audio generation).
2.  **Model Settings**: Select your preferred text (LLM) and audio (TTS) models. Configure hardware profiles and other advanced settings.
3.  **Run Pipeline**: Start the process. LlamaNote will generate output files in the `output/` directory.
