# LlamaNote GPU Optimization Guide

This guide explains the advanced GPU optimization features in LlamaNote, designed to help you run large language models (LLMs) on consumer-grade hardware, including GPUs with as little as 4GB of VRAM.

## Table of Contents

1. [The Challenge: Running Large Models](#the-challenge-running-large-models)
2. [Core Optimization Strategies](#core-optimization-strategies)
   - [Quantization](#quantization)
   - [CPU Offloading (Layer Splitting)](#cpu-offloading-layer-splitting)
   - [Advanced KV-Cache Management](#advanced-kv-cache-management)
3. [Key Menu: Advanced Memory Optimization](#key-menu-advanced-memory-optimization)
4. [Feature Deep Dive](#feature-deep-dive)
   - [Auto OOM Handling](#auto-oom-handling)
   - [Iterative Layer Splitting](#iterative-layer-splitting)
   - [Advanced Cache System (Hot/Cold + Sliding Window)](#advanced-cache-system-hotcold--sliding-window)
   - [Cache Strategies](#cache-strategies)
5. [Practical Scenarios & Recommendations](#practical-scenarios--recommendations)
   - [Scenario 1: 4GB VRAM GPU (e.g., GTX 1650)](#scenario-1-4gb-vram-gpu-eg-gtx-1650)
   - [Scenario 2: 8GB VRAM GPU (e.g., RTX 3070)](#scenario-2-8gb-vram-gpu-eg-rtx-3070)
   - [Scenario 3: CPU-Only Inference](#scenario-3-cpu-only-inference)
6. [Troubleshooting](#troubleshooting)

---

## The Challenge: Running Large Models

Modern LLMs are powerful but require significant memory. A 7-billion-parameter model can consume over 28GB of memory in full precision. This is far beyond the capacity of most consumer GPUs.

LlamaNote employs several state-of-the-art techniques to overcome this limitation.

## Core Optimization Strategies

### 1. Quantization

-   **What it is**: Reducing the precision of the model's weights (parameters) from 32-bit or 16-bit floating-point numbers to lower-precision integers (e.g., 8-bit or 4-bit).
-   **Impact**: Drastically reduces the model's memory footprint.
    -   **16-bit (Half Precision)**: ~14GB for a 7B model.
    -   **8-bit**: ~7GB for a 7B model.
    -   **4-bit**: ~3.5-4GB for a 7B model.
-   **How to use**: Set via **Model Settings -> Set Local Hardware Profile**.

### 2. CPU Offloading (Layer Splitting)

-   **What it is**: Loading only a portion of the model's layers onto the GPU's VRAM, while keeping the rest in your computer's system RAM. During inference, the data flows between the CPU and GPU as needed.
-   **Impact**: Allows you to run models that are larger than your VRAM by leveraging system RAM.
-   **How to use**: Enabled by default. The number of layers on the GPU is determined by your **Hardware Profile** or can be set manually in the **Advanced Memory Optimization** menu.

### 3. Advanced KV-Cache Management

-   **What it is**: The KV-cache is the memory of the conversation, storing attention keys and values for previously generated tokens. It can grow very large. LlamaNote's advanced cache system manages this intelligently.
-   **Impact**: Prevents out-of-memory errors during long conversations or when processing large documents.
-   **How to use**: Enabled by default via the **Advanced Cache System** toggle.

## Key Menu: Advanced Memory Optimization

This menu (**Model Settings -> Advanced Memory Optimization**) is the central hub for controlling these features. Here's a breakdown of the most important settings:

-   **CPU Offloading**: Master switch for layer splitting.
-   **Auto OOM Handling**: The most important feature. Automatically recovers from CUDA Out-of-Memory (OOM) errors by offloading more layers to the CPU.
-   **Advanced Cache System**: Enables the integrated hot/cold cache and sliding window attention.
-   **Cache Strategy**: Balances memory savings vs. quality (`aggressive`, `balanced`, `quality`).
-   **Reset to Recommended Defaults**: A safe starting point for a 4GB VRAM system.

## Feature Deep Dive

### Auto OOM Handling

This is LlamaNote's safety net. If the model requires more VRAM than available, a CUDA OOM error occurs. Instead of crashing, LlamaNote will:

1.  Catch the error.
2.  Reduce the number of layers on the GPU (offloading more to the CPU).
3.  Retry the operation.

This process repeats until the model fits in memory or it determines that CPU-only inference is required.

### Iterative Layer Splitting

When Auto OOM Handling is triggered, this setting determines *how* layers are offloaded.

-   **Enabled (Default)**: Offloads layers one by one. This is slower but finds the absolute maximum number of layers that can fit on the GPU.
-   **Disabled**: Offloads a fixed percentage of layers (e.g., 20%). This is faster but may offload more than necessary.

For most users, the default enabled setting is recommended.

### Advanced Cache System (Hot/Cold + Sliding Window)

This is a unified system that combines two powerful techniques:

1.  **Hybrid (Hot/Cold) KV-Cache**:
    -   The most recently used parts of the KV-cache ("hot" cache) are kept in fast VRAM.
    -   Older parts of the cache ("cold" cache) are moved to slower system RAM.
    -   This is like the virtual memory system in an operating system, but for the LLM's conversational memory.

2.  **Sliding Window Attention**:
    -   Instead of allowing the model to "see" every single previous token (which causes the KV-cache to grow indefinitely), it only pays attention to a recent "window" of tokens.
    -   To preserve long-range context, a small number of initial tokens (the "prefix") are always kept in the window.

By combining these, LlamaNote can handle very long contexts while keeping memory usage bounded and predictable.

### Cache Strategies

The **Cache Strategy** setting provides simple presets for the advanced cache system:

-   **Aggressive**:
    -   **Window**: 1024 tokens
    -   **Hot Cache**: 256 MB
    -   **Use for**: Maximum memory savings. Good for tasks that don't require long-range context. May slightly reduce quality.

-   **Balanced (Default)**:
    -   **Window**: 2048 tokens
    -   **Hot Cache**: 512 MB
    -   **Use for**: The best trade-off between memory usage and quality. Recommended for most users and tasks.

-   **Quality**:
    -   **Window**: 4096 tokens
    -   **Hot Cache**: 1024 MB
    -   **Use for**: Prioritizing quality and long-range context, assuming you have sufficient VRAM (e.g., > 6GB).

## Practical Scenarios & Recommendations

### Scenario 1: 4GB VRAM GPU (e.g., GTX 1650)

-   **Hardware Profile**: `low_vram` or `minimal_vram`.
-   **Quantization**: Must use `4bit`.
-   **Advanced Settings**:
    -   Use the **Recommended Defaults**.
    -   **Cache Strategy**: Start with `balanced`. If you still encounter OOM errors, switch to `aggressive`.
    -   Ensure **Auto OOM Handling** is `ENABLED`.

### Scenario 2: 8GB VRAM GPU (e.g., RTX 3070)

-   **Hardware Profile**: `medium_vram`.
-   **Quantization**: `4bit` is still recommended for speed, but `8bit` is feasible.
-   **Advanced Settings**:
    -   **Cache Strategy**: `balanced` or `quality`.
    -   You have more headroom, so you can afford a larger hot cache and window size for better performance on long documents.

### Scenario 3: CPU-Only Inference

-   If you have no compatible GPU, LlamaNote will run entirely on the CPU.
-   **Hardware Profile**: Select a profile, but GPU-specific settings will be ignored.
-   **Performance**: Expect significantly slower generation speeds. For CPU-only use, GGUF models are often faster than Transformers models.

## Troubleshooting

-   **Problem**: I get a CUDA Out of Memory error, and the program crashes.
    -   **Solution**: Ensure **Auto OOM Handling** is `ENABLED` in the Advanced Memory Optimization menu. If it is, and it still crashes, your system may not have enough *total* memory (VRAM + RAM) to run the model, even with offloading.

-   **Problem**: Generation is very slow.
    -   **Solution**: This is expected if many layers are offloaded to the CPU. You can see the final device map in the logs after the model loads. The more layers on "cpu", the slower it will be. Consider using a smaller model or a more aggressive quantization method (e.g., switch from 8-bit to 4-bit).

-   **Problem**: The model's responses seem to lose context or get repetitive on long documents.
    -   **Solution**: Switch the **Cache Strategy** to `balanced` or `quality`. This increases the sliding window size, allowing the model to "remember" more of the preceding text.