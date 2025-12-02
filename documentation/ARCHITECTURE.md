# LlamaNote Architecture Documentation

**Version:** 2.0
**Last Updated:** 2025-11-10
**Total Files:** 107 Python files (~40,000 lines of code)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Module Breakdown](#module-breakdown)
4. [Data Flow & Pipeline](#data-flow--pipeline)
5. [Class Hierarchies](#class-hierarchies)
6. [Reusable Components Catalog](#reusable-components-catalog)
7. [Code Duplication Analysis](#code-duplication-analysis)
8. [Integration Points](#integration-points)
9. [Memory Management System](#memory-management-system)
10. [Extension & Customization](#extension--customization)

---

## System Overview

### Purpose

LlamaNote is a sophisticated document processing pipeline that converts PDFs and text files into various output formats (podcasts, technical documentation, narratives) using Large Language Models (LLMs) and Text-to-Speech (TTS) engines. It's designed for efficiency on low-resource hardware (4GB VRAM) through advanced memory management and layer splitting.

### Key Features

- **Multi-backend support**: Local HuggingFace models, GGUF quantized models, cloud APIs (OpenAI, Google, Anthropic)
- **Advanced memory management**: Dynamic layer splitting, KV-cache optimization, OOM recovery
- **Checkpoint system**: Resume processing from any stage with configuration restoration
- **Multi-speaker TTS**: Dataset-based, audio extraction, or random speaker embedding generation
- **Progress tracking**: Hierarchical progress bars using Rich library
- **Batch processing**: Dynamic batch sizing with OOM detection and adjustment
- **Model lifecycle management**: Auto-load/unload models based on pipeline stage requirements

### Technology Stack

- **Core**: Python 3.8+
- **ML Frameworks**: PyTorch, Transformers (HuggingFace)
- **Audio**: SoundFile, Librosa, Pydub
- **PDF Processing**: PyPDF2, pdfplumber, pymupdf
- **UI**: Rich (terminal), Questionary (menus)
- **Serialization**: Pickle (checkpoints), JSON (metadata, config)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLI / Menu Interface                       │
│                      (cli.py, menu.py, menu_checkpoint.py)          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      Configuration Manager                          │
│               (config/manager.py, config/settings.py)               │
│                 ┌─────────────┬─────────────┬───────────┐           │
│                 │  Profiles   │   Presets   │   Keys    │           │
│                 └─────────────┴─────────────┴───────────┘           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                       Processing Pipeline                           │
│                      (core/pipeline.py)                             │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  ┌─────────┐  ┌──────────┐  │
│  │ Extract │→ │Preprocess│→ │ Chunk │→ │ Process │→ │  Format  │  │
│  └─────────┘  └──────────┘  └───────┘  └─────────┘  └──────────┘  │
│       │                                      │             │         │
│       ▼                                      ▼             ▼         │
│  ┌─────────┐                          ┌─────────┐  ┌──────────┐    │
│  │  Audio  │                          │ LLM     │  │  File    │    │
│  │  Stage  │                          │ Backend │  │  Handler │    │
│  └─────────┘                          └─────────┘  └──────────┘    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐   ┌────────▼────────┐   ┌───────▼──────┐
│  Checkpoint     │   │  Model          │   │  Memory      │
│  Manager        │   │  Lifecycle      │   │  Manager     │
│  (io/)          │   │  Manager        │   │  (utils/)    │
└─────────────────┘   └─────────────────┘   └──────────────┘
```

---

## Module Breakdown

### 1. Core Modules (`src/core/`)

#### **pipeline.py** (1,303 lines)
- **Purpose**: Orchestrates the entire processing workflow
- **Key Classes**: `ProcessingPipeline`
- **Responsibilities**:
  - Stage execution and coordination
  - Backend injection and management
  - Checkpoint integration
  - Progress tracking
  - Error handling and retry logic

#### **types.py** (416 lines)
- **Purpose**: Central type definitions and data structures
- **Key Classes**:
  - `PipelineConfig`: Complete pipeline configuration
  - `GenerationResult`: Standardized LLM output
  - `AudioConfig`, `AudioResult`: Audio-specific configuration and results
  - `ExtractionResult`, `ChunkingResult`, `FilterResult`: Stage results
  - `QuantizationConfig`, `LayerSplitConfig`: Hardware configuration
- **Enums**: `ProcessingMode`, `ChunkingStrategy`

#### **stages.py**
- **Purpose**: Stage execution logic
- **Key Classes**: `PipelineStageExecutor`
- **Responsibilities**:
  - Extract, preprocess, chunk, process, filter, format, save stages
  - Error handling per stage
  - Stage-specific data transformations

#### **errors.py**
- **Purpose**: Custom exception hierarchy
- **Classes**: `PipelineError`, `ModelLoadError`, `FileProcessingError`, `GenerationError`, `MissingDataError`

#### **model_lifecycle_manager.py**
- **Purpose**: Dynamic model loading/unloading
- **Key Classes**: `ModelLifecycleManager`
- **Responsibilities**:
  - Load models only when needed
  - Unload models after use to free memory
  - Track memory freed
  - Statistics tracking

#### **stage_analyzer.py**
- **Purpose**: Analyzes stage dependencies and creates execution plans
- **Key Classes**: `StageAnalyzer`, `ModelLifecyclePlan`
- **Responsibilities**:
  - Determine which models are needed for which stages
  - Create optimal load/unload schedule
  - Memory usage estimation

---

### 2. Models Module (`src/models/`)

#### **backends/** (8 implementations)

**base.py** (337 lines)
- Abstract base classes: `LLMBackend`, `AudioBackend`
- Common methods: `load()`, `unload()`, `generate()`, `process_with_chat_template()`
- Error handling: `_build_error_result()`

**local_hf.py** (845 lines) - Largest backend
- HuggingFace Transformers backend
- Features:
  - `LocalModelLoader`: Helper for model loading with quantization
  - Dynamic cache integration (removed custom implementation, uses transformers' `DynamicCache`)
  - Layer splitting support
  - OOM recovery
  - Batch inference support via `_generate_batch_request()`

**local_gguf.py**
- GGUF/llama.cpp backend
- Supports CPU/GPU layer splitting via `n_gpu_layers`

**local_audio.py** (1,287 lines) - Largest file
- Local TTS backend (SpeechT5, VibeVoice)
- Multi-speaker support
- Speaker embedding management
- Audio concatenation and post-processing

**batch.py** (307 lines)
- Batch processing coordinator
- Dynamic batch sizing with OOM handling
- Checkpoint callback support
- Resume from chunk functionality

**dynamic_batch.py**
- Dynamic batch size adjustment
- OOM detection and reduction
- Success tracking and optimization

**Cloud backends**:
- `openai_llm.py`, `openai_audio.py`: OpenAI API
- `anthropic_llm.py`: Claude API
- `google_llm.py`: Gemini API

**mapper.py**
- Backend factory pattern
- Maps provider IDs to backend classes

#### **registry.py** (474 lines)
- **Purpose**: Model database management
- **Key Classes**: `ModelEntry`, `ModelRegistry`
- **Features**:
  - Thread-safe singleton
  - Predefined models from JSON
  - HuggingFace Hub integration
  - Model caching tracking
  - Search and filter capabilities

#### **hyperparameters.py** (650 lines)
- **Purpose**: LLM generation parameters management
- **Key Classes**: `HyperparameterConfig`, `HyperparameterPresetManager`
- **Features**:
  - Preset management (creative, balanced, precise)
  - Parameter validation
  - Per-model optimization hints

#### **speaker_embeddings/** (4 modules)
- **manager.py** (521 lines): Orchestrates embedding generation
- **dataset_sampler.py**: Samples from HuggingFace datasets
- **audio_extractor.py** (617 lines): Extracts embeddings from audio files
- **random_generator.py**: Generates random embeddings
- **base.py**: Abstract base for embedding strategies

#### **vibevoice/** (Microsoft's TTS model integration)
- Modular implementation with custom diffusion heads
- Processor and tokenizer implementations
- Scheduler (DPM solver) - 1,064 lines

#### **cache/** (Advanced caching - deprecated in favor of transformers' built-in)
- `transformers_cache_impl.py` (620 lines): Custom cache implementation
- `model_adapters.py` (693 lines): Attention layer patching
- **Note**: Advanced cache system removed due to performance issues

---

### 3. Processing Module (`src/processing/`)

#### **text_chunker.py** (528 lines)
- **Key Classes**: `TextChunker`
- **Strategies**:
  - Word boundary
  - Sentence boundary
  - Paragraph boundary
  - Semantic chunking
  - Sliding window
  - Token-based
- **Features**: Overlap handling, chunk merging, size validation

#### **text_preprocessor.py**
- **Key Classes**: `TextPreprocessor`
- **Features**:
  - URL/email removal
  - Encoding fixes
  - Whitespace normalization
  - Audio-specific cleanup
  - Sentence segmentation (optional)

#### **pdf_extractor.py**
- **Key Classes**: `PDFProcessor`
- **Features**:
  - Multiple extraction backends (PyPDF2, pdfplumber, pymupdf)
  - Metadata extraction
  - Layout preservation
  - OCR support (via pymupdf)

#### **response_filter.py**
- **Key Classes**: `ResponseFilter`, `ChunkedResponseFilter`
- **Features**:
  - Thinking token removal (`<think>`, `<|thinking|>`, etc.)
  - Acknowledgment removal ("Sure, I'll help...")
  - Instruction regurgitation cleanup
  - Artifact token removal
  - Model-specific pattern support
  - Statistical tracking

#### **audio_processor.py**
- **Purpose**: Audio post-processing
- **Features**: Volume normalization, speed adjustment, pitch shift

---

### 4. IO Module (`src/io/`)

#### **checkpoints.py** (2,219 lines) - Largest file
- **Key Classes**: `CheckpointManager`
- **Features**:
  - Advanced checkpoint management with gzip compression
  - Stage-aware compatibility checking
  - Fuzzy configuration matching (allow minor differences)
  - Audio array compression (FLAC)
  - Hash-based organization
  - Descriptive checkpoint naming
  - Metadata generation (`.meta.json` sidecar files)
  - Archival milestones (20%, 40%, 60%, 80%, 100%)

#### **checkpoint_registry.py** (702 lines)
- **Key Classes**: `CheckpointRegistry`
- **Features**:
  - CSV-based checkpoint tracking
  - Fast search and filtering
  - Startup verification
  - Cleanup of stale entries

#### **checkpoint_migration.py**
- **Purpose**: Migrate legacy checkpoints to new format
- **Features**: Batch migration, preserve original option

#### **checkpoint_startup.py**
- **Purpose**: Startup helper for checkpoint system
- **Features**: Verify registry, offer metadata regeneration

#### **model_abbreviations.py**
- **Key Classes**: `ModelAbbreviationManager`
- **Purpose**: Shorten model names for checkpoint filenames
- **Features**: Configurable mappings, automatic fallback patterns

#### **file_handler.py**
- **Key Classes**: `FileHandler`
- **Purpose**: File I/O operations
- **Features**: Timestamped outputs, metadata inclusion, format conversion

#### **batch_manager.py**
- **Purpose**: Manage batch file processing
- **Features**: Queue management, parallel processing coordination

#### **directory_manager.py**
- **Purpose**: Directory structure management
- **Features**: Output organization, temporary file handling

#### **session_manager.py**
- **Purpose**: Session state persistence
- **Features**: Save/restore processing sessions

#### **text_file_manager.py**
- **Purpose**: Text file operations
- **Features**: Read, write, format conversion

#### **report_generator.py**
- **Purpose**: Generate processing reports
- **Features**: Statistics, timings, error summaries

#### **layer_split_cache.py**
- **Purpose**: Cache optimal layer split configurations
- **Features**: Fast lookup, validation tracking

---

### 5. Formatting Module (`src/formatting/`)

#### **base_formatter.py** (175 lines)
- **Key Classes**: `BaseFormatter`, `PatternMatcher`, `PatternReplacer`
- **Purpose**: Base class and factory for formatters
- **Features**: Code block preservation, document structure

#### **podcast_formatter.py**
- **Purpose**: Format output as podcast dialogue
- **Features**: Speaker detection, emotion tags, scene markers

#### **technical_formatter.py**
- **Purpose**: Format output as technical documentation
- **Features**: Code block formatting, heading hierarchy, tables

#### **narrative_formatter.py**
- **Purpose**: Format output as narrative text
- **Features**: Paragraph flow, descriptive style

---

### 6. Configuration Module (`src/config/`)

#### **settings.py** (500+ lines)
- **Purpose**: Central configuration constants
- **Contents**:
  - File paths and directories
  - Default values
  - System prompts (preprocessing, podcast planning, generation)
  - Markdown styles
  - Stage weights for progress tracking
  - Model cache settings
  - Pipeline stages

#### **manager.py**
- **Key Classes**: `ConfigManager`
- **Purpose**: Runtime configuration management
- **Features**:
  - Profile management (save/load/list)
  - Preset management
  - Validation
  - Directory management

#### **profiles.py**
- **Purpose**: User profile definitions
- **Features**: Profile serialization, validation

#### **presets.py**
- **Purpose**: Predefined configuration presets
- **Features**: Hyperparameter presets, GPU presets

#### **gpu_presets.py**
- **Purpose**: Hardware-specific presets
- **Features**: 4GB, 8GB, 12GB+ configurations

#### **cloud_keys.py**
- **Purpose**: API key management
- **Features**: Secure storage, environment variable support

---

### 7. Utils Module (`src/utils/`)

#### **logger.py** (600+ lines)
- **Key Classes**: `MemoryMonitor`, `LoggingProgress`, `DualProgressTracker`, `ConsoleOutput`
- **Purpose**: Centralized logging and monitoring
- **Features**:
  - Multi-handler logging (console, file, error)
  - Memory monitoring (RAM, VRAM)
  - Progress tracking
  - Error context saving

#### **progress_tracking.py** (344 lines)
- **Key Classes**: `ProgressManager`
- **Purpose**: Hierarchical progress tracking using Rich library
- **Features**:
  - Pipeline-level progress (weighted by stage)
  - Stage-level progress (chunks)
  - Batch-level progress (individual batches)
  - Checkpoint notification display
  - Thread-safe updates

#### **memory_manager.py** (800+ lines)
- **Key Classes**: `CUDAMemoryManager`, `OOMRecoveryStrategy`
- **Purpose**: Advanced memory management
- **Features**:
  - OOM detection and recovery
  - Progressive layer offloading
  - Cache management
  - Memory statistics

#### **device_map_builder.py**
- **Purpose**: Build device maps for model layer distribution
- **Features**: GPU/CPU/disk allocation, memory estimation

#### **auto_layer_split.py**
- **Purpose**: Automatically discover optimal layer splits
- **Features**: Binary search for max GPU layers, validation

#### **layer_split_finder.py** (586 lines)
- **Purpose**: Interactive layer split discovery tool
- **Features**: Testing, validation, caching

#### **memory_estimator.py**
- **Purpose**: Estimate model memory requirements
- **Features**: Parameter counting, quantization effects, KV-cache estimation

#### **helpers.py**
- **Purpose**: Utility functions
- **Functions**: Token estimation, resource cleanup, device management, type conversions

#### **decorators.py**
- **Purpose**: Reusable decorators
- **Decorators**: `@log_execution_time`, `@log_resource_usage`, `@retry_on_error`

#### **validators.py**
- **Purpose**: Input validation
- **Functions**: Path validation, config validation, parameter validation

#### **file_browser.py**
- **Purpose**: Interactive file selection
- **Features**: Directory browsing, file filtering

#### **lru_cache_manager.py**
- **Purpose**: Manage LRU caches
- **Features**: Cache size limits, eviction policies

---

### 8. Compression Module (`src/compression/`)

#### **compress_audio.py**
- **Purpose**: Audio compression for checkpoints
- **Features**: FLAC compression, quality preservation

#### **compress_text.py**
- **Purpose**: Text compression
- **Features**: Gzip compression, deduplication

#### **compression_config.py**
- **Purpose**: Compression settings
- **Features**: Compression levels, format selection

---

### 9. Menu Module (`src/menu.py`, `src/menu_checkpoint.py`)

#### **menu.py** (2,169 lines) - Second largest file
- **Purpose**: Interactive CLI menu system
- **Features**:
  - Model selection and configuration
  - Pipeline execution
  - Checkpoint management
  - Settings management
  - Multi-level navigation

#### **menu_checkpoint.py** (609 lines)
- **Purpose**: Checkpoint-specific menu
- **Features**: View, resume, delete, migrate checkpoints

---

## Data Flow & Pipeline

### Pipeline Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                           │
│    - Load configuration                                     │
│    - Initialize components (PDF processor, chunker, etc.)   │
│    - Inject LLM and Audio backends                          │
│    - Initialize checkpoint manager                          │
│    - Create model lifecycle plan                            │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 2. CHECKPOINT RESUME CHECK                                  │
│    - Search for compatible checkpoints                      │
│    - Check configuration compatibility                      │
│    - Load checkpoint if found and compatible                │
│    - Calculate remaining stages                             │
│    - Restore configuration from checkpoint                  │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 3. EXTRACT STAGE                                            │
│    - Read input file (PDF, TXT, MD)                         │
│    - Extract text and metadata                              │
│    - Multiple backend support (PyPDF2, pdfplumber, pymupdf) │
│    - Save checkpoint                                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 4. PREPROCESS STAGE                                         │
│    - Clean text (remove artifacts, fix encoding)            │
│    - Normalize whitespace                                   │
│    - Apply mode-specific preprocessing                      │
│    - Save checkpoint                                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 5. CHUNK STAGE                                              │
│    - Split text into chunks using selected strategy         │
│    - Apply overlap if configured                            │
│    - Validate chunk sizes                                   │
│    - Save checkpoint                                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 6. PROCESS STAGE (LLM Generation)                          │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Model Lifecycle Management                          │ │
│    │ - Unload audio model if loaded                      │ │
│    │ - Load text model if not loaded                     │ │
│    └─────────────────────────────────────────────────────┘ │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Batch Processing Loop                               │ │
│    │ For each batch of chunks:                           │ │
│    │   - Get dynamic batch size                          │ │
│    │   - Try parallel batch inference                    │ │
│    │   - Fallback to sequential on error                 │ │
│    │   - Handle OOM with batch size reduction            │ │
│    │   - Save checkpoint every N chunks                  │ │
│    │   - Update progress tracking                        │ │
│    └─────────────────────────────────────────────────────┘ │
│    - Concatenate results                                    │
│    - Save stage checkpoint                                  │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 7. FILTER STAGE                                             │
│    - Remove thinking tokens                                 │
│    - Remove acknowledgments                                 │
│    - Clean up instruction regurgitation                     │
│    - Remove artifact tokens                                 │
│    - Track statistics                                       │
│    - Save checkpoint                                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 8. FORMAT STAGE                                             │
│    - Apply mode-specific formatting (podcast/tech/narrative)│
│    - Preserve code blocks                                   │
│    - Add document structure                                 │
│    - Save checkpoint                                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 9. SAVE STAGE                                               │
│    - Write formatted text to file                           │
│    - Include metadata if configured                         │
│    - Generate timestamp if configured                       │
│    - Save checkpoint                                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 10. AUDIO STAGE (if enabled)                               │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Model Lifecycle Management                          │ │
│    │ - Unload text model if loaded                       │ │
│    │ - Load audio model if not loaded                    │ │
│    └─────────────────────────────────────────────────────┘ │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Speaker Embedding Management                        │ │
│    │ - Detect speakers in text                           │ │
│    │ - Generate embeddings (dataset/audio/random)        │ │
│    │ - Cache embeddings for reuse                        │ │
│    └─────────────────────────────────────────────────────┘ │
│    ┌─────────────────────────────────────────────────────┐ │
│    │ Audio Generation Loop                               │ │
│    │ For each text chunk:                                │ │
│    │   - Select speaker embedding                        │ │
│    │   - Generate audio                                  │ │
│    │   - Concatenate audio arrays                        │ │
│    │   - Save checkpoint every N chunks                  │ │
│    │   - Clear CUDA cache if configured                  │ │
│    └─────────────────────────────────────────────────────┘ │
│    - Post-process audio (normalize, speed, pitch)           │
│    - Save audio file                                        │
│    - Save checkpoint                                        │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│ 11. CLEANUP                                                 │
│    - Unload all models                                      │
│    - Clear CUDA cache                                       │
│    - Generate processing report                             │
│    - Return PipelineResult                                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Transformation Flow

```
Input File (PDF/TXT)
    │
    ▼ (Extract)
Raw Text + Metadata (ExtractionResult)
    │
    ▼ (Preprocess)
Cleaned Text (str)
    │
    ▼ (Chunk)
Text Chunks (ChunkingResult: List[TextChunk])
    │
    ▼ (Process - LLM)
Generated Chunks (List[GenerationResult])
    │
    ▼ (Filter)
Filtered Chunks (List[FilterResult])
    │
    ▼ (Format)
Formatted Text (str)
    │
    ▼ (Save)
Output File (.md/.txt/.json)
    │
    ▼ (Audio - optional)
Audio File (.wav/.mp3/.flac)
```

---

## Class Hierarchies

### Backend Hierarchy

```
┌─────────────┐
│ LLMBackend  │ (Abstract Base Class)
└──────┬──────┘
       │
       ├── LocalHFBackend (HuggingFace Transformers)
       ├── LocalGGUFBackend (llama.cpp)
       ├── OpenAIBackend (GPT-4, GPT-3.5)
       ├── AnthropicBackend (Claude)
       └── GoogleBackend (Gemini)

┌───────────────┐
│ AudioBackend  │ (Abstract Base Class)
└───────┬───────┘
        │
        ├── LocalAudioBackend (SpeechT5, VibeVoice)
        └── OpenAIAudioBackend (TTS-1)
```

### Formatter Hierarchy

```
┌──────────────────┐
│ BaseFormatter    │
└────────┬─────────┘
         │
         ├── PodcastFormatter
         ├── TechnicalFormatter
         └── NarrativeFormatter
```

### Embedding Strategy Hierarchy

```
┌────────────────────────────┐
│ BaseSpeakerEmbeddingSource │ (Abstract)
└──────────┬─────────────────┘
           │
           ├── DatasetSampler
           ├── AudioExtractor
           └── RandomGenerator
```

### Error Hierarchy

```
┌──────────────┐
│PipelineError │ (Base Exception)
└──────┬───────┘
       │
       ├── ModelLoadError
       ├── FileProcessingError
       ├── GenerationError
       └── MissingDataError
```

---

## Reusable Components Catalog

### 1. Progress Tracking (`utils/progress_tracking.py`)

**Component**: `ProgressManager`

**Reusability**: High
**Dependencies**: Rich library

**Features**:
- Hierarchical progress bars (pipeline → stage → batch)
- Weighted progress calculation
- Thread-safe updates
- Checkpoint notification display
- Context manager support

**Usage Pattern**:
```python
with ProgressManager(stage_weights) as progress:
    progress.add_pipeline_progress(total_weight)
    progress.add_stage_progress("process", total_chunks)
    for chunk in chunks:
        # Process chunk
        progress.update_stage("process", completed)
```

**Extraction Potential**: Standalone package for hierarchical progress tracking

---

### 2. Memory Management (`utils/memory_manager.py`)

**Component**: `CUDAMemoryManager`, `OOMRecoveryStrategy`

**Reusability**: High
**Dependencies**: PyTorch

**Features**:
- Memory statistics (allocated, reserved, peak)
- Cache clearing (normal and aggressive)
- OOM detection
- Progressive recovery (cache clearing → layer offloading)
- Memory string parsing ("4GB", "2048MB", "4GiB")

**Usage Pattern**:
```python
manager = CUDAMemoryManager()
stats = manager.get_memory_stats()
if manager.is_oom_error(exception):
    manager.clear_cache(aggressive=True)
```

**Extraction Potential**: Standalone package for PyTorch memory management

---

### 3. Logging System (`utils/logger.py`)

**Component**: `LoggingProgress`, `DualProgressTracker`, `MemoryMonitor`, `ConsoleOutput`

**Reusability**: High
**Dependencies**: Standard library, psutil, torch (optional)

**Features**:
- Multi-handler logging (console, file, error)
- Context-aware logging
- Memory monitoring (RAM and VRAM)
- Progress tracking integration
- Error context saving

**Usage Pattern**:
```python
logger = get_logger_conf(__name__)
monitor = MemoryMonitor(logger)
monitor.start()
with LoggingProgress(logger, "Task", total) as progress:
    for item in items:
        # Process
        progress.update(1)
monitor.check("After processing")
```

**Extraction Potential**: Standalone logging package with progress and memory tracking

---

### 4. Response Filtering (`processing/response_filter.py`)

**Component**: `ResponseFilter`, `ChunkedResponseFilter`

**Reusability**: High
**Dependencies**: Standard library (re)

**Features**:
- Pattern-based filtering (thinking tokens, acknowledgments)
- Model-specific pattern support
- Statistical tracking
- Batch processing support
- Configurable patterns

**Usage Pattern**:
```python
filter = ResponseFilter(model_config)
result = filter.filter(text, remove_thinking=True)
print(f"Removed {result.filter_stats['thinking_segments']} thinking segments")
```

**Extraction Potential**: Standalone LLM output cleaning package

---

### 5. Batch Processing (`models/backends/batch.py`)

**Component**: `BatchProcessor`, `DynamicBatchManager`

**Reusability**: High
**Dependencies**: PyTorch, custom backend interface

**Features**:
- Dynamic batch sizing
- OOM handling with automatic reduction
- Checkpoint callback support
- Resume from chunk
- Sequential and parallel processing
- Progress tracking integration

**Usage Pattern**:
```python
processor = BatchProcessor(backend, batch_size=4)
results = processor.process_batch(
    texts, system_prompt,
    checkpoint_callback=save_checkpoint,
    checkpoint_interval=10
)
```

**Extraction Potential**: Generic batch processing framework with OOM handling

---

### 6. Checkpoint Management (`io/checkpoints.py`)

**Component**: `CheckpointManager`

**Reusability**: High
**Dependencies**: Standard library (pickle, gzip, json)

**Features**:
- Compression (gzip for data, FLAC for audio)
- Stage-aware compatibility
- Fuzzy configuration matching
- Metadata generation
- Hash-based organization
- Descriptive naming

**Usage Pattern**:
```python
manager = CheckpointManager(base_dir)
manager.save(input_path, config, stage, data)
checkpoint = manager.get_resume_checkpoint(input_path, config)
```

**Extraction Potential**: Generic checkpoint system for long-running pipelines

---

### 7. Pattern Matching (`formatting/base_formatter.py`)

**Component**: `PatternMatcher`, `PatternReplacer`

**Reusability**: High
**Dependencies**: Standard library (re)

**Features**:
- Compiled pattern caching
- Score-based detection
- Batch replacements
- Error handling for invalid patterns

**Usage Pattern**:
```python
matcher = PatternMatcher()
mode = matcher.detect_from_patterns(text, patterns_map)

replacer = PatternReplacer(patterns)
cleaned = replacer.replace(text)
```

**Extraction Potential**: Standalone text pattern matching/replacement utility

---

### 8. Model Registry (`models/registry.py`)

**Component**: `ModelRegistry`, `ModelEntry`

**Reusability**: High
**Dependencies**: Standard library (json, threading)

**Features**:
- Thread-safe singleton
- JSON persistence
- HuggingFace Hub integration
- Predefined models
- Search and filter
- Cache tracking

**Usage Pattern**:
```python
registry = get_registry()
entry = registry.find_entry("qwen3-4b")
models = registry.list_models(is_predefined=True, sort_by="downloads")
```

**Extraction Potential**: Generic model/resource registry system

---

### 9. Configuration Management (`config/manager.py`)

**Component**: `ConfigManager`

**Reusability**: High
**Dependencies**: Standard library (json, pathlib)

**Features**:
- Profile management (CRUD)
- Preset management
- Directory management
- Validation
- Thread-safe operations

**Usage Pattern**:
```python
manager = ConfigManager()
manager.save_profile("my_profile", config)
config = manager.load_profile("my_profile")
profiles = manager.list_profiles()
```

**Extraction Potential**: Generic configuration management system

---

### 10. Device Map Builder (`utils/device_map_builder.py`)

**Component**: `LayerSplitConfigManager`, `DeviceMapBuilder`

**Reusability**: Medium
**Dependencies**: PyTorch, HuggingFace Transformers

**Features**:
- Layer distribution across devices
- Memory estimation
- Configuration caching
- Validation tracking

**Usage Pattern**:
```python
manager = LayerSplitConfigManager()
saved_split = manager.load_config(model_id, quant, gpu_memory_gb)
device_map = saved_split.device_map
```

**Extraction Potential**: Model layer distribution utility for HuggingFace models

---

## Code Duplication Analysis

### Areas of Duplication

#### 1. **Text Chunking Logic**

**Locations**:
- `processing/text_chunker.py`: Full implementation
- `models/backends/base.py` (`AudioBackend._split_text()`): Simplified chunking

**Analysis**: Audio backend duplicates basic word-boundary chunking. Could use `TextChunker` directly or create a shared utility.

**Recommendation**:
```python
# Refactor AudioBackend to use TextChunker
from ...processing.text_chunker import TextChunker

class AudioBackend:
    def _split_text(self, text: str) -> List[str]:
        chunker = TextChunker(target_size=self.config.chunk_size)
        result = chunker.chunk_text(text)
        return [chunk.text for chunk in result.chunks]
```

---

#### 2. **Memory Statistics Collection**

**Locations**:
- `utils/memory_manager.py` (`CUDAMemoryManager.get_memory_stats()`)
- `utils/logger.py` (`MemoryMonitor._get_gpu_usage_bytes()`, `_get_ram_usage_bytes()`)

**Analysis**: Both collect similar memory statistics. Could consolidate into single source.

**Recommendation**:
```python
# Create shared memory utility
# utils/memory_stats.py
class MemoryStats:
    @staticmethod
    def get_gpu_stats() -> Dict[str, float]:
        # Consolidated implementation
        pass

    @staticmethod
    def get_ram_stats() -> Dict[str, float]:
        # Consolidated implementation
        pass

# Then use in both CUDAMemoryManager and MemoryMonitor
```

---

#### 3. **Progress Tracking**

**Locations**:
- `utils/logger.py`: `LoggingProgress`, `DualProgressTracker`
- `utils/progress_tracking.py`: `ProgressManager`

**Analysis**: Legacy progress tracking in logger.py, new Rich-based tracking in progress_tracking.py. Should consolidate to single system.

**Recommendation**: Deprecate legacy progress tracking, use `ProgressManager` everywhere.

---

#### 4. **Configuration Validation**

**Locations**:
- `core/types.py`: `__post_init__` methods in dataclasses
- `utils/validators.py`: Validation functions
- `config/manager.py`: Profile/preset validation

**Analysis**: Validation logic scattered across multiple locations.

**Recommendation**: Centralize validation in `validators.py`:
```python
# utils/validators.py
class ConfigValidator:
    @staticmethod
    def validate_pipeline_config(config: PipelineConfig) -> List[str]:
        # All validation logic here
        pass

    @staticmethod
    def validate_audio_config(config: AudioConfig) -> List[str]:
        pass

# Use in dataclasses
@dataclass
class PipelineConfig:
    def __post_init__(self):
        errors = ConfigValidator.validate_pipeline_config(self)
        if errors:
            raise ValueError(f"Invalid config: {errors}")
```

---

#### 5. **File Operations**

**Locations**:
- `io/file_handler.py`: Main file operations
- `io/text_file_manager.py`: Text-specific operations
- `io/directory_manager.py`: Directory operations
- `config/manager.py`: Profile file operations

**Analysis**: File I/O patterns repeated across modules.

**Recommendation**: Create base file manager class:
```python
# io/base_file_manager.py
class BaseFileManager:
    def read_json(self, path: Path) -> Dict:
        pass

    def write_json(self, path: Path, data: Dict):
        pass

    def atomic_write(self, path: Path, content: str):
        # Write to temp, then rename (atomic)
        pass

# Then inherit in specific managers
class FileHandler(BaseFileManager):
    pass

class TextFileManager(BaseFileManager):
    pass
```

---

#### 6. **Error Handling Patterns**

**Locations**:
- Repeated try-except blocks across all backend implementations
- Similar error result creation

**Analysis**: Could use decorators or wrapper functions.

**Recommendation**:
```python
# utils/decorators.py
def handle_backend_errors(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"{func.__name__} failed: {e}")
            return self._build_error_result(e, duration)
    return wrapper

# Use in backends
class LocalHFBackend(LLMBackend):
    @handle_backend_errors
    def _generate_request(self, prompt, hyperparams):
        # Just the core logic, error handling automatic
        pass
```

---

#### 7. **Checkpoint Callback Patterns**

**Locations**:
- `core/pipeline.py`: Multiple checkpoint save callbacks
- `models/backends/batch.py`: Checkpoint callback in batch processor

**Analysis**: Similar checkpoint callback patterns repeated.

**Recommendation**: Create checkpoint callback builder:
```python
# io/checkpoint_callbacks.py
class CheckpointCallbackBuilder:
    @staticmethod
    def create_process_callback(manager, input_path, config):
        def callback(chunk_idx, results, extra_data):
            # Standard checkpoint save logic
            pass
        return callback

    @staticmethod
    def create_audio_callback(manager, input_path, config):
        def callback(chunk_idx, audio_arrays, extra_data):
            # Audio-specific checkpoint save
            pass
        return callback
```

---

### Opportunities for Wrapper Classes

#### 1. **Model Loading Wrapper**

**Current**: Each backend reimplements model loading with quantization, device maps, etc.

**Proposed**:
```python
# models/loader.py
class ModelLoader:
    """Unified model loading with quantization and device mapping."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_hf_model(self, model_id: str):
        # Centralized HF model loading
        # Handles quantization, device maps, layer splitting
        pass

    def load_gguf_model(self, model_path: Path):
        # Centralized GGUF loading
        pass

# Use in backends
class LocalHFBackend(LLMBackend):
    def load(self):
        loader = ModelLoader(self.config)
        self.model, self.tokenizer = loader.load_hf_model(self.model_specifier)
```

---

#### 2. **Generation Wrapper**

**Current**: Each backend implements similar generation flow (preprocessing, generation, post-processing).

**Proposed**:
```python
# models/generation_wrapper.py
class GenerationWrapper:
    """Wraps generation with common pre/post processing."""

    def __init__(self, backend: LLMBackend, filter: ResponseFilter):
        self.backend = backend
        self.filter = filter

    def generate_with_filtering(self, prompt, hyperparams, remove_thinking=True):
        # Generate
        result = self.backend.generate(prompt, hyperparams)
        # Filter
        if remove_thinking:
            filtered = self.filter.filter(result.raw_output)
            result.filtered_output = filtered.filtered_text
        return result
```

---

#### 3. **Audio Processing Wrapper**

**Current**: Audio post-processing repeated in audio backend and audio processor.

**Proposed**:
```python
# processing/audio_wrapper.py
class AudioProcessingWrapper:
    """Wraps audio generation with post-processing."""

    def __init__(self, backend: AudioBackend, config: AudioConfig):
        self.backend = backend
        self.config = config

    def generate_with_postprocessing(self, text, output_path):
        # Generate audio
        result = self.backend.generate_audio(text, output_path)

        # Post-process
        if self.config.volume_normalize:
            self._normalize_volume(result.audio_path)
        if self.config.speed != 1.0:
            self._adjust_speed(result.audio_path)

        return result
```

---

## Integration Points

### External Service Integrations

1. **HuggingFace Hub** (`models/hub.py`)
   - Model search and download
   - Model information retrieval
   - Authentication

2. **OpenAI API** (`models/backends/openai_llm.py`, `openai_audio.py`)
   - GPT-4, GPT-3.5 text generation
   - TTS audio generation

3. **Anthropic API** (`models/backends/anthropic_llm.py`)
   - Claude text generation

4. **Google API** (`models/backends/google_llm.py`)
   - Gemini text generation

### File Format Support

1. **PDF**
   - PyPDF2 (basic extraction)
   - pdfplumber (layout-aware extraction)
   - pymupdf (OCR support)

2. **Audio**
   - WAV (input/output)
   - MP3 (output)
   - FLAC (output, checkpoint compression)

3. **Text**
   - TXT (input/output)
   - Markdown (input/output)
   - JSON (output, metadata)
   - HTML (output)

### Model Format Support

1. **PyTorch/Transformers**
   - Full precision (FP32)
   - Half precision (FP16, BF16)
   - 8-bit quantization (bitsandbytes)
   - 4-bit quantization (bitsandbytes)

2. **GGUF**
   - Various quantization formats (Q4_K_M, Q5_K_M, Q8_0, etc.)
   - CPU/GPU layer splitting

---

## Memory Management System

### Layer Splitting Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Model Layers                            │
├────────────────────────────────────────────────────────────┤
│ Layer 0  │ GPU (VRAM)                                      │
│ Layer 1  │ GPU (VRAM)                                      │
│ ...      │ GPU (VRAM)                                      │
│ Layer N  │ GPU (VRAM)     ← Split Point                    │
├──────────┼─────────────────────────────────────────────────┤
│ Layer N+1│ CPU (RAM)                                       │
│ ...      │ CPU (RAM)                                       │
│ Layer M  │ CPU (RAM)                                       │
├──────────┼─────────────────────────────────────────────────┤
│ Layer M+1│ Disk (Offload) ← If needed                      │
│ ...      │ Disk (Offload)                                  │
└────────────────────────────────────────────────────────────┘
```

### OOM Recovery Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. OOM Detected                                         │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ 2. Clear CUDA Cache (aggressive=False)                  │
│    - torch.cuda.empty_cache()                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ Success? ──────────────────────┐
                 │                                │
                 │ No                             │ Yes
                 │                                │
┌────────────────▼────────────────────┐          │
│ 3. Clear CUDA Cache (aggressive=True)│          │
│    - torch.cuda.synchronize()       │          │
│    - gc.collect()                   │          │
│    - torch.cuda.empty_cache()       │          │
└────────────────┬────────────────────┘          │
                 │                                │
                 │ Success? ──────────────────────┤
                 │                                │
                 │ No                             │
                 │                                │
┌────────────────▼────────────────────┐          │
│ 4. Offload KV Cache to CPU          │          │
└────────────────┬────────────────────┘          │
                 │                                │
                 │ Success? ──────────────────────┤
                 │                                │
                 │ No                             │
                 │                                │
┌────────────────▼────────────────────┐          │
│ 5. Offload Context to CPU           │          │
└────────────────┬────────────────────┘          │
                 │                                │
                 │ Success? ──────────────────────┤
                 │                                │
                 │ No                             │
                 │                                │
┌────────────────▼────────────────────┐          │
│ 6. Progressive Layer Offloading     │          │
│    - Calculate layers to offload    │          │
│    - Rebuild device map             │          │
│    - Reload model with new map      │          │
└────────────────┬────────────────────┘          │
                 │                                │
                 │ Success? ──────────────────────┤
                 │                                │
                 │ No                             │ Yes
                 │                                │
┌────────────────▼────────────────────┐   ┌──────▼───────┐
│ 7. FAILURE                          │   │   RESUME     │
│    - Return error                   │   │ PROCESSING   │
└─────────────────────────────────────┘   └──────────────┘
```

### Dynamic Batch Sizing

```
Initial batch_size = N

Loop:
    Try processing with current batch_size

    If OOM:
        batch_size = max(1, batch_size // 2)
        Clear cache
        Retry

    If success:
        Record success (batch_size, peak_memory)

        If peak_memory < threshold and batch_size < max:
            batch_size = min(max_batch_size, batch_size + 1)
```

---

## Extension & Customization

### Adding New Backends

#### LLM Backend

1. Create new file in `models/backends/`
2. Inherit from `LLMBackend`
3. Implement required methods:
   - `load()`
   - `unload()`
   - `_generate_request()`
   - `_chat_request()`
4. Register in `mapper.py`

```python
# models/backends/my_backend.py
from .base import LLMBackend

class MyBackend(LLMBackend):
    def load(self) -> bool:
        # Load your model
        return True

    def unload(self):
        # Clean up
        pass

    def _generate_request(self, prompt, hyperparams):
        # Generate text
        return GenerationResult(...)

    def _chat_request(self, system_prompt, user_message, hyperparams):
        # Generate with chat template
        return GenerationResult(...)

# models/backends/mapper.py
def get_backend_class(provider_id: str):
    mapping = {
        # ... existing mappings
        "my_provider": MyBackend,
    }
    return mapping.get(provider_id)
```

#### Audio Backend

Similar process, inherit from `AudioBackend`, implement:
- `load()`
- `unload()`
- `generate_audio()`

---

### Adding New Processing Modes

1. Add new mode to `ProcessingMode` enum in `core/types.py`
2. Create new formatter in `formatting/`
3. Add mode-specific prompt in `config/settings.py`
4. Add mode to formatter factory in `formatting/base_formatter.py`

```python
# core/types.py
class ProcessingMode(Enum):
    # ... existing
    INTERVIEW = "interview"

# formatting/interview_formatter.py
class InterviewFormatter(BaseFormatter):
    def _apply_format(self, text, **kwargs):
        # Format as interview Q&A
        pass

# formatting/base_formatter.py
def get_formatter(mode: ProcessingMode, style_name: Optional[str] = None):
    # ... existing
    if mode == ProcessingMode.INTERVIEW:
        return InterviewFormatter()

# config/settings.py
INTERVIEW_PROMPT = """
Create an interview-style dialogue...
"""
```

---

### Adding New Chunking Strategies

1. Add strategy to `ChunkingStrategy` enum in `core/types.py`
2. Implement strategy in `processing/text_chunker.py`

```python
# processing/text_chunker.py
class TextChunker:
    def chunk_text(self, text: str, strategy: Optional[ChunkingStrategy] = None):
        # ... existing
        elif strategy == ChunkingStrategy.MY_STRATEGY:
            chunks = self._chunk_my_way(text)
        # ...

    def _chunk_my_way(self, text: str) -> List[TextChunk]:
        # Implement your chunking logic
        pass
```

---

### Adding New Filter Patterns

1. Add patterns to `processing/response_filter.py`
2. Optionally add model-specific patterns in model registry

```python
# processing/response_filter.py
NEW_PATTERNS = [
    (r"<my_tag>(.*?)</my_tag>", re.DOTALL),
]

class ResponseFilter:
    def __init__(self, model_config):
        # ... existing
        for pattern, flags in NEW_PATTERNS:
            self.patterns.append((re.compile(pattern, flags), ""))
```

---

### Configuration Extensions

#### Custom Profiles

```python
# Via ConfigManager
manager = ConfigManager()
custom_config = PipelineConfig(
    mode=ProcessingMode.CUSTOM,
    model_specifier="my-model",
    # ... custom settings
)
manager.save_profile("my_profile", custom_config)
```

#### Custom Presets

```python
# config/presets.py
CUSTOM_PRESET = {
    "name": "my_preset",
    "temperature": 0.8,
    "top_p": 0.95,
    # ...
}

# Add to PRESET_DEFINITIONS
PRESET_DEFINITIONS = {
    # ... existing
    "my_preset": CUSTOM_PRESET,
}
```

---

## Best Practices & Guidelines

### For Extending LlamaNote

1. **Use Existing Base Classes**: Inherit from `LLMBackend`, `AudioBackend`, `BaseFormatter`, etc.
2. **Follow Type Hints**: Use type annotations for better IDE support and documentation
3. **Use Dataclasses**: For configuration and result objects
4. **Leverage Decorators**: Use `@log_execution_time`, `@log_resource_usage` for consistency
5. **Error Handling**: Always use custom exception classes from `core/errors.py`
6. **Logging**: Use `get_logger_conf(__name__)` for module-specific loggers
7. **Configuration**: Store settings in `config/settings.py`, not hardcoded
8. **Testing**: Add unit tests for new components
9. **Documentation**: Add docstrings and update this architecture doc

### For Code Reuse

1. **Check Existing Utilities**: Before writing new code, check `utils/` for existing implementations
2. **Extract Common Patterns**: If you see duplication, extract to utility function
3. **Use Wrappers**: For cross-cutting concerns (logging, error handling, retries)
4. **Avoid Circular Imports**: Use TYPE_CHECKING, lazy imports, or refactor
5. **Thread Safety**: Use locks for shared state, consider singleton patterns

---

## Performance Optimization Notes

### Current Optimizations

1. **Memory Management**:
   - Layer splitting (GPU/CPU/disk)
   - Dynamic batch sizing
   - OOM recovery
   - Cache clearing between chunks

2. **Checkpointing**:
   - Gzip compression (~60% reduction)
   - FLAC audio compression (~50% reduction)
   - Incremental saves (every N chunks)

3. **Model Loading**:
   - Lazy loading (only when needed)
   - Auto-unload (when no longer needed)
   - Quantization (4bit, 8bit, 16bit)

4. **Progress Tracking**:
   - Weighted progress (accurate time estimates)
   - Off-screen rendering (minimal overhead)

### Potential Optimizations

1. **Caching**:
   - LRU cache for processed chunks
   - Embedding cache (already implemented)
   - Compiled regex patterns (already implemented)

2. **Parallelization**:
   - True parallel batch inference (currently sequential batches)
   - Multi-GPU support
   - Distributed processing

3. **I/O**:
   - Asynchronous file I/O
   - Streaming audio generation
   - Background checkpoint saves

4. **Algorithms**:
   - Faster chunking (C extension?)
   - Optimized filter patterns
   - Semantic chunking with embeddings

---

## Dependency Graph

### Critical Dependencies

```
Core Pipeline
    ├── Core Types (types.py)
    ├── Config Manager
    │   ├── Settings
    │   └── Profiles/Presets
    ├── Processing
    │   ├── PDF Extractor
    │   ├── Text Preprocessor
    │   ├── Text Chunker
    │   └── Response Filter
    ├── Formatters
    │   └── Base Formatter
    ├── File Handler
    ├── Checkpoint Manager
    │   ├── Model Abbreviations
    │   └── Checkpoint Registry
    └── Model Lifecycle Manager
        └── Stage Analyzer

LLM Backend (injected)
    ├── Model Registry
    ├── Hyperparameters
    ├── Memory Manager
    ├── Batch Processor
    │   └── Dynamic Batch Manager
    └── Backend Implementation
        ├── Local HF
        ├── Local GGUF
        └── Cloud APIs

Audio Backend (injected)
    ├── Speaker Embeddings
    │   ├── Manager
    │   ├── Dataset Sampler
    │   ├── Audio Extractor
    │   └── Random Generator
    └── Backend Implementation
        ├── Local Audio
        └── OpenAI Audio

Utilities (used everywhere)
    ├── Logger
    ├── Progress Manager
    ├── Memory Manager
    ├── Validators
    ├── Helpers
    └── Decorators
```

---

## File Size Distribution

### Top 20 Largest Files

| File | Lines | Purpose |
|------|-------|---------|
| `io/checkpoints.py` | 2,219 | Checkpoint management |
| `menu.py` | 2,169 | Interactive menu system |
| `core/pipeline.py` | 1,303 | Pipeline orchestration |
| `models/backends/local_audio.py` | 1,287 | Local TTS backend |
| `models/vibevoice/modular/modular_vibevoice_tokenizer.py` | 1,194 | VibeVoice tokenizer |
| `models/vibevoice/schedule/dpm_solver.py` | 1,064 | DPM solver for diffusion |
| `models/backends/local_hf.py` | 845 | HuggingFace backend |
| `models/vibevoice/modular/modeling_vibevoice_inference.py` | 780 | VibeVoice inference |
| `io/checkpoint_registry.py` | 702 | Checkpoint registry |
| `models/cache/model_adapters.py` | 693 | Attention patching (deprecated) |
| `models/vibevoice/processor/vibevoice_processor.py` | 686 | VibeVoice processor |
| `cli.py` | 678 | CLI interface |
| `models/hyperparameters.py` | 650 | Hyperparameter management |
| `models/cache/transformers_cache_impl.py` | 620 | Custom cache (deprecated) |
| `models/speaker_embeddings/audio_extractor.py` | 617 | Audio embedding extraction |
| `menu_checkpoint.py` | 609 | Checkpoint menu |
| `utils/layer_split_finder.py` | 586 | Layer split discovery |
| `processing/text_chunker.py` | 528 | Text chunking |
| `models/speaker_embeddings/manager.py` | 521 | Embedding management |
| `models/registry.py` | 474 | Model registry |

**Total**: 39,685 lines across 107 files

---

## Conclusion

LlamaNote is a sophisticated, modular document processing pipeline with extensive features for low-resource hardware. The architecture emphasizes:

1. **Modularity**: Clear separation of concerns, pluggable backends
2. **Reusability**: Many components can be extracted as standalone packages
3. **Extensibility**: Easy to add new backends, modes, strategies
4. **Robustness**: Comprehensive error handling, checkpointing, recovery
5. **Performance**: Memory management, batch processing, caching
6. **User Experience**: Progress tracking, interactive menus, detailed logging

The codebase is well-structured but has opportunities for consolidation (progress tracking, memory utilities, validation). The reusable components catalog identifies high-value extractions for standalone packages.

---

**Document Maintainer**: Claude Code
**Next Review Date**: 2025-12-01
**Questions**: See CLAUDE.md for development guidance
