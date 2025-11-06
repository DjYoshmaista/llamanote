# LlamaNote Architecture Documentation Index

## Complete Documentation Set for Model Loading & Memory Management

This documentation set provides comprehensive coverage of how LlamaNote loads models, manages memory, and executes the processing pipeline.

### Documents in This Set

#### 1. **ARCHITECTURE_MODEL_MEMORY.md** (Comprehensive Reference)
   - **Size:** 15 KB | **Lines:** 430
   - **Best for:** Deep understanding, implementation details, decision rationale
   - **Contains:**
     - Text model (LLM) loading & pipeline integration
     - Audio model (TTS) loading & pipeline integration
     - Model offloading/unloading mechanisms
     - KV-cache and context window handling
     - Pipeline stage execution flow
     - Settings/configuration system
     - Summary table of all components
     - List of current features and limitations
   
   **Read this when:** You need to understand the complete architecture or modify core components

#### 2. **ARCHITECTURE_DIAGRAMS.txt** (Visual Reference)
   - **Size:** 11 KB | **Lines:** 351
   - **Best for:** Visual learners, quick understanding of flows
   - **Contains:**
     - Model lifecycle diagram (CLI to memory)
     - Pipeline stage execution with model loading
     - Batch processing with unloading flow
     - Memory management layers visualization
     - Configuration hierarchy diagram
     - KV-cache and context window handling flow
     - Backend factory pattern diagram
   
   **Read this when:** You need to visualize how components interact

#### 3. **QUICK_REFERENCE_MODEL_MEMORY.md** (Practical Guide)
   - **Size:** 9 KB | **Lines:** 304
   - **Best for:** Quick lookup, debugging, configuration
   - **Contains:**
     - File locations reference table
     - Key code snippets with line numbers
     - Configuration paths and CLI arguments
     - Model loading flow overview
     - Architecture decision rationale
     - Debugging guide for memory issues
     - Best practices for different hardware
     - Pipeline stages quick reference
     - Common issues and solutions
   
   **Read this when:** You need to quickly find something or debug a problem

### Quick Navigation

#### By Use Case

**I want to understand how models are loaded:**
1. Start: ARCHITECTURE_DIAGRAMS.txt (Model Lifecycle)
2. Deep dive: ARCHITECTURE_MODEL_MEMORY.md (Section 1-2)
3. Code details: QUICK_REFERENCE_MODEL_MEMORY.md (Key Code Snippets)

**I want to understand memory management:**
1. Start: ARCHITECTURE_DIAGRAMS.txt (Memory Management Layers)
2. Deep dive: ARCHITECTURE_MODEL_MEMORY.md (Section 3-4)
3. Debugging: QUICK_REFERENCE_MODEL_MEMORY.md (Debugging Guide)

**I need to configure memory for my hardware:**
1. Start: QUICK_REFERENCE_MODEL_MEMORY.md (Configuration Best Practices)
2. Deep dive: ARCHITECTURE_MODEL_MEMORY.md (Section 6)
3. CLI guide: QUICK_REFERENCE_MODEL_MEMORY.md (CLI Arguments)

**I'm debugging a memory issue:**
1. Start: QUICK_REFERENCE_MODEL_MEMORY.md (Debugging Memory Issues)
2. Reference: ARCHITECTURE_MODEL_MEMORY.md (Key Features & Gaps)
3. Common fixes: QUICK_REFERENCE_MODEL_MEMORY.md (Common Issues Table)

**I'm modifying the code:**
1. Start: QUICK_REFERENCE_MODEL_MEMORY.md (File Locations Reference)
2. Understand: ARCHITECTURE_DIAGRAMS.txt (relevant flow)
3. Implementation: ARCHITECTURE_MODEL_MEMORY.md (detailed section)

#### By Component

**CLI & Entry Point:**
- QUICK_REFERENCE_MODEL_MEMORY.md → File Locations (cli.py)
- ARCHITECTURE_MODEL_MEMORY.md → Section 1-2 (Loading Points)

**Pipeline Execution:**
- QUICK_REFERENCE_MODEL_MEMORY.md → Pipeline Stages Quick Reference
- ARCHITECTURE_DIAGRAMS.txt → Section 2 (Pipeline Stage Execution)
- ARCHITECTURE_MODEL_MEMORY.md → Section 5 (Pipeline Stage Execution)

**Backend System:**
- QUICK_REFERENCE_MODEL_MEMORY.md → File Locations (models/backends/)
- ARCHITECTURE_DIAGRAMS.txt → Section 7 (Backend Factory Pattern)
- ARCHITECTURE_MODEL_MEMORY.md → Section 1-2 (Backend Architecture)

**Memory Management:**
- QUICK_REFERENCE_MODEL_MEMORY.md → Configuration Best Practices
- ARCHITECTURE_DIAGRAMS.txt → Section 4 (Memory Management Layers)
- ARCHITECTURE_MODEL_MEMORY.md → Section 3-4 (Offloading & KV Cache)

**Configuration System:**
- QUICK_REFERENCE_MODEL_MEMORY.md → Configuration Paths & CLI Arguments
- ARCHITECTURE_MODEL_MEMORY.md → Section 6 (Settings/Configuration)
- ARCHITECTURE_DIAGRAMS.txt → Section 5 (Configuration Hierarchy)

### Key Code Locations Quick Link

| Task | File | Function/Class | Line |
|------|------|-----------------|------|
| Load text model | cli.py | run_cli_processing() | 568 |
| Load audio model | cli.py | run_cli_processing() | 585 |
| Backend factory | models/backends/__init__.py | get_llm_backend() | 79 |
| LocalHF loading | models/backends/local_hf.py | LocalHFBackend.load() | 322 |
| Memory config | cli.py | run_cli_processing() | 490 |
| Pipeline execution | core/pipeline.py | ProcessingPipeline.process_file() | 131 |
| Process stage (LLM) | core/pipeline.py | process stage | 276 |
| Audio stage | core/pipeline.py | audio stage | 455 |
| Model unload | cli.py | finally block | 670 |
| Memory cleanup | core/stages.py | per-chunk cleanup | 215 |

### Documentation Maintenance

**Last Updated:** 2025-11-03
**Coverage:** Complete as of branch 0.0.21
**Related Files:**
- ARCHITECTURE.md (older, high-level overview)

### How to Use This Documentation

1. **For Overview:** Start with ARCHITECTURE_DIAGRAMS.txt
2. **For Details:** Refer to ARCHITECTURE_MODEL_MEMORY.md
3. **For Quick Lookup:** Use QUICK_REFERENCE_MODEL_MEMORY.md
4. **For Code Changes:** Cross-reference with file locations table

### Scope

This documentation covers:
- Model loading lifecycle (CLI to in-memory execution)
- Memory management strategies and configurations
- Pipeline stage execution and dependencies
- Backend factory and provider patterns
- Configuration hierarchy and settings
- KV-cache and context window management
- Checkpoint and resume mechanisms

### Not Covered

For other aspects of LlamaNote, see:
- Text processing: README.md or src/processing/
- File I/O: src/io/ directory
- Formatting: src/formatting/ directory
- Logging: src/utils/logger.py

---

**Start reading:** Pick a document above based on your use case!
