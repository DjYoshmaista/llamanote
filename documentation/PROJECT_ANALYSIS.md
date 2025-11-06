# LlamaNote Project Analysis

## 1. High-Level Architecture

LlamaNote is a Python-based command-line application designed to process documents (PDFs, text files) and generate structured notes, summaries, or high-quality audio outputs. It employs a modular, pipeline-based architecture that allows for flexible and extensible processing.

The core of the application is a `ProcessingPipeline` that executes a series of configurable stages. Each stage is responsible for a specific task, such as text extraction, preprocessing, chunking, language model processing, and audio generation. This design allows users to customize the pipeline to suit their specific needs, running only the stages they require.

The application supports a variety of Large Language Models (LLMs) and Text-to-Speech (TTS) models through a system of backends. This includes local models (via Hugging Face Transformers and GGUF) and cloud-based services (OpenAI, Anthropic, Google). A `ModelRegistry` and `ModelHub` facilitate the management, discovery, and downloading of models.

A key focus of the architecture is efficient memory management, enabling the use of large models on consumer-grade hardware. This is achieved through a combination of quantization, CPU offloading (layer splitting), and an advanced caching system.

## 2. Core Capabilities

- **Document Processing:** Ingests PDF, TXT, and MD files, extracting and cleaning text content.
- **Flexible LLM Integration:** Supports a wide range of local and cloud-based LLMs.
- **High-Quality Audio Generation:** Converts text to speech using local or cloud-based TTS models.
- **Multi-Speaker TTS:** Can generate audio with distinct voices for different speakers identified in the text.
- **Advanced GPU Optimization:** Employs quantization, layer splitting, and a custom KV-cache to run large models on GPUs with as little as 4GB of VRAM.
- **Interactive CLI:** A comprehensive menu system allows for easy configuration of all processing aspects.
- **Checkpoint/Resume:** Automatically saves progress and can resume long-running tasks, preventing loss of work.
- **Extensible Pipeline:** The processing pipeline is modular, allowing for the addition of new stages and formatters.

## 3. Code Structure

The project's source code is organized within the `src/` directory, with a clear separation of concerns:

-   **`src/cli.py`, `src/menu.py`:** Entry points and user interface for the application.
-   **`src/core/`:** Contains the central `ProcessingPipeline`, type definitions (`types.py`), and stage execution logic.
-   **`src/config/`:** Manages application settings, memory profiles, and API keys.
-   **`src/models/`:** A comprehensive package for all model-related functionality.
    -   **`backends/`:** Concrete implementations for different LLM and TTS providers (local and cloud).
    -   **`cache/`:** Advanced caching mechanisms for memory optimization.
    -   **`speaker_embeddings/`:** Manages the generation and lifecycle of speaker embeddings for multi-speaker TTS.
    -   **`registry.py`, `hub.py`:** Model discovery, registration, and management.
-   **`src/processing/`:** Houses the individual processing components like `PDFProcessor`, `TextChunker`, and `AudioPostProcessor`.
-   **`src/io/`:** Handles all file system interactions, including checkpointing and output generation.
-   **`src/formatting/`:** Contains formatters for structuring the final output (e.g., podcast script, technical document).
-   **`src/utils/`:** A collection of helper modules for logging, validation, decorators, and memory management.

## 4. Data Flow

1.  **Input:** The user selects one or more documents through the CLI or file browser.
2.  **Extraction:** The `PDFProcessor` (or a simple file reader) extracts the raw text.
3.  **Preprocessing:** The `TextPreprocessor` cleans the text, removing artifacts and normalizing whitespace.
4.  **Chunking:** The `TextChunker` splits the text into manageable chunks based on the selected strategy.
5.  **Processing (LLM):** The `BatchProcessor` sends each chunk to the configured `LLMBackend` (e.g., `LocalHFBackend`). The LLM processes the text based on the system prompt and hyperparameters.
6.  **Filtering:** The `ResponseFilter` cleans the LLM's output, removing any "thinking" tags or other artifacts.
7.  **Formatting:** The appropriate `Formatter` (e.g., `PodcastFormatter`) structures the text into its final form.
8.  **Saving:** The `FileHandler` saves the final text output to a file.
9.  **Audio Generation (Optional):** If enabled, the `AudioBackend` synthesizes the final text into an audio file, potentially using the `SpeakerEmbeddingManager` to assign different voices.

Throughout this process, the `CheckpointManager` saves the state at the end of each stage, allowing the pipeline to be resumed if interrupted.

## 5. Configuration System

Configuration is managed hierarchically:

-   **`src/config/settings.py`:** Contains hardcoded default values and constants.
-   **`src/config/profiles.py`:** Defines memory profiles (`low_vram`, `medium_vram`, etc.) that bundle together quantization and layer splitting settings.
-   **`src/config/presets.py`:** Manages hyperparameter presets for different generation styles.
-   **`src/config/manager.py`:** The `ConfigManager` provides a unified interface for accessing all configurations.
-   **`src/core/types.py`:** Dataclasses like `PipelineConfig`, `QuantizationConfig`, and `LayerSplitConfig` hold the runtime configuration.
-   **Interactive Menu:** The menu system in `src/menu.py` allows the user to override any of the default settings.

## 6. Model Management

-   **`ModelRegistry`:** A singleton class that maintains a database of all known models, both predefined and discovered.
-   **`ModelHub`:** An interface for interacting with the Hugging Face Hub to search for and download new models.
-   **Backends (`src/models/backends/`):** Each backend is a self-contained class responsible for loading, running, and unloading a specific type of model (e.g., `LocalHFBackend` for local Transformers models, `OpenAIBackend` for the OpenAI API).
-   **`ModelLifecycleManager`:** This component, guided by the `StageAnalyzer`, dynamically loads and unloads models as needed during the pipeline execution to minimize peak memory usage.

## 7. Memory Management

LlamaNote employs a multi-pronged approach to memory optimization:

-   **Quantization:** Models can be loaded in 4-bit or 8-bit precision to drastically reduce their memory footprint.
-   **Layer Splitting (CPU Offloading):** For models that don't fit entirely in VRAM, layers can be split between the GPU and CPU. The `DeviceMapBuilder` and `LayerSplitFinder` help automate the process of finding the optimal split.
-   **OOM Recovery:** The `OOMRecoveryStrategy` in `src/utils/memory_manager.py` provides a safety net. If a CUDA Out-of-Memory error occurs, it progressively offloads more layers to the CPU and retries, allowing the model to load successfully, albeit with a performance trade-off.
-   **Advanced Caching (`src/models/cache/`):** A custom `LlamaNoteDynamicCache` implements a hybrid KV-cache with a sliding window. This allows for the processing of very long documents by intelligently paging the model's conversational memory (the KV-cache) between the GPU (hot cache) and CPU (cold cache).
-   **Model Lifecycle Management:** By only loading models right before the stage that needs them and unloading them immediately after, the application avoids holding multiple large models in memory simultaneously.

## 8. Extensibility

The architecture is designed to be extensible:

-   **New Models:** Adding a new local model is as simple as adding its ID to the model registry. Adding support for a new cloud provider involves creating a new backend class that inherits from `LLMBackend`.
-   **New Formatters:** To support a new output format, a new class inheriting from `BaseFormatter` can be created in the `src/formatting/` directory.
-   **New Pipeline Stages:** A new processing stage can be added by creating a function for it in `src/core/stages.py` and inserting it into the `DEFAULT_PIPELINE_STAGES` list.

## 9. State of the Project

Based on the `GEMINI.md`, `CLAUDE.md`, and `feature-parity-todo.md` documents, the project is in an advanced state of development, with a focus on achieving feature parity with a more mature `llamanote-github` repository.

-   **Completed:**
    -   The core pipeline architecture.
    -   Advanced GPU optimization features (OOM handling, layer splitting).
    -   A sophisticated, production-ready caching system.
    -   The foundational infrastructure for the multi-speaker TTS system.
    -   A robust checkpoint/resume system.
-   **In Progress / To-Do:**
    -   Full integration of the multi-speaker TTS system into the pipeline and menu.
    -   Implementation of voice cloning from audio files.
    -   Enhancements to the menu system, including a file browser and dedicated checkpoint management section.
-   **Future Enhancements:**
    -   Distributed checkpoint support.
    -   Adaptive compression for checkpoints.
    -   Performance profiling and multi-GPU support.
