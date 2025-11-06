# LlamaNote Gemini - Feature Parity TODO List

This document outlines the tasks required to bring the `llamanote-backup` repository to feature parity with the `llamanote-github` repository.

## Phase 1: Multi-Speaker TTS System Implementation

### 1.1. Implement the Speaker Embedding Manager
- **Task:** Create the `src/models/speaker_embeddings/manager.py` file.
- **Details:** Implement the `SpeakerEmbeddingManager` class to handle the lifecycle of speaker embeddings, including generation, caching, and persistence.

### 1.2. Implement the Audio Embedding Extractor
- **Task:** Create the `src/models/speaker_embeddings/audio_extractor.py` file.
- **Details:** Implement the `AudioEmbeddingExtractor` class for voice cloning from audio files. This will require integrating with the SpeechBrain library.

### 1.3. Integrate the Speaker Embedding System into the Menu
- **Task:** Update the `src/menu.py` file.
- **Details:** Add a submenu for configuring the multi-speaker TTS system, including options for generation method, gender filtering, and cache management.

### 1.4. Integrate the Speaker Embedding System into the Pipeline
- **Task:** Update the `src/core/pipeline.py` and `src/models/backends/local_audio.py` files.
- **Details:** Integrate the multi-speaker TTS system into the main pipeline, with automatic speaker detection and voice assignment.

## Phase 2: Advanced Caching System Implementation

### 2.1. Implement the Dynamic Cache
- **Task:** Create the `src/models/cache/transformers_cache_impl.py` file.
- **Details:** Implement the `LlamaNoteDynamicCache` class, which provides a custom implementation of the Hugging Face Transformers Cache interface.

### 2.2. Implement Model Adapters
- **Task:** Create the `src/models/cache/model_adapters.py` file.
- **Details:** Implement model-specific adapters to apply architecture-specific optimizations for different model families.

### 2.3. Implement the Generation Wrapper
- **Task:** Create the `src/models/cache/generation_wrapper.py` file.
- **Details:** Implement the `CachedGenerationWrapper` class to wrap the model's `generate` method and integrate the custom cache.

### 2.4. Integrate the Caching System into the Pipeline
- **Task:** Update the `src/core/pipeline.py` and relevant backend files.
- **Details:** Integrate the advanced caching system into the main pipeline to improve memory management and performance.

## Phase 3: Menu and User Experience Enhancements

### 3.1. Implement the File Browser
- **Task:** Update the `src/menu.py` file.
- **Details:** Add a file browser for easier input file selection.

### 3.2. Implement Checkpoint Management
- **Task:** Update the `src/menu.py` file.
- **Details:** Add a dedicated section for managing pipeline checkpoints.

## Phase 4: Documentation and Finalization

### 4.1. Update Documentation
- **Task:** Create the `PHASE` and `TODO` documents in the `documentation` directory.
- **Details:** Create detailed documentation that tracks the project's development and future plans.

### 4.2. Generate Project Analysis
- **Task:** Create the `documentation/PROJECT_ANALYSIS.md` file.
- **Details:** Generate a comprehensive report on the project's architecture, structure, and capabilities. This document will be used to inform future development and provide a high-level overview of the project.

### 4.3. Final Review and Testing
- **Task:** Perform a final review of the codebase and test all new features.
- **Details:** Ensure that all new features are working as expected and that there are no regressions.
