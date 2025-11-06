# LlamaNote Speaker Embedding Guide

This guide provides a comprehensive overview of the multi-speaker Text-to-Speech (TTS) system in LlamaNote, powered by speaker embeddings. This system allows you to generate audio with distinct, consistent voices for different speakers in your text.

## Table of Contents

1. [What are Speaker Embeddings?](#what-are-speaker-embeddings)
2. [How it Works in LlamaNote](#how-it-works-in-llamanote)
3. [Formatting Your Text for Multi-Speaker TTS](#formatting-your-text-for-multi-speaker-tts)
4. [Configuration Options](#configuration-options)
   - [Enabling Multi-Speaker Mode](#enabling-multi-speaker-mode)
   - [Generation Methods](#generation-methods)
   - [Voice Customization](#voice-customization)
5. [Embedding Generation Methods in Detail](#embedding-generation-methods-in-detail)
   - [Random (Default)](#random-default)
   - [Dataset](#dataset)
   - [Audio (Voice Cloning)](#audio-voice-cloning)
6. [Caching and Persistence](#caching-and-persistence)
7. [Troubleshooting](#troubleshooting)

---

## What are Speaker Embeddings?

A **speaker embedding** is a vector of numbers that captures the unique characteristics of a person's voice. Think of it as a "voice fingerprint." TTS models like SpeechT5 can use these embeddings to condition their output, producing speech that sounds like the person the embedding represents.

LlamaNote's multi-speaker system manages these embeddings, assigning a unique one to each speaker detected in your document.

## How it Works in LlamaNote

The process is fully automated:

1.  **Speaker Detection**: LlamaNote scans your formatted text for speaker labels (e.g., `[Speaker Host]`).
2.  **Voice Assignment**: For each unique speaker name, the `SpeakerEmbeddingManager` assigns a unique voice (embedding).
    - If a voice has been previously generated for that speaker, it's loaded from the cache.
    - If it's a new speaker, a new voice is generated using the configured method (e.g., `random`, `dataset`).
3.  **Segmented Generation**: The text is split into segments, one for each speaker.
4.  **Audio Synthesis**: Each segment is converted to audio using the corresponding speaker's voice embedding.
5.  **Concatenation**: All audio segments are seamlessly stitched together into a single audio file.

This ensures that "Speaker Host" always sounds like "Speaker Host," and "Speaker Guest" always has their own distinct voice.

## Formatting Your Text for Multi-Speaker TTS

To use the multi-speaker feature, you must format your text with clear speaker labels. The required format is:

```markdown
**[Speaker NAME]:** Text spoken by the speaker...
```

**Key Formatting Rules:**

-   The label must start with `**[Speaker`.
-   The speaker's name can contain letters, numbers, and spaces.
-   The label must end with `]:**`.
-   The text for that speaker follows immediately on the same line.

**Example:**

```markdown
**[Speaker Host]:** Welcome to the LlamaNote podcast! Today, we're discussing the future of local AI.

**[Speaker Dr. Anya Sharma]:** It's a pleasure to be here. The advancements in model quantization have been revolutionary.

**[Speaker Host]:** Absolutely. Models that once required a datacenter can now run on a 4GB GPU. How does the new caching system play into this?

**[Speaker Dr. Anya Sharma]:** The hybrid KV-cache is a game-changer for managing long contexts on limited hardware. It intelligently pages data between GPU and CPU memory.
```

In this example, LlamaNote will create two distinct voices: one for "Host" and one for "Dr. Anya Sharma."

## Configuration Options

You can configure the speaker embedding system from the **Model Settings -> Speaker Embeddings** menu.

### Enabling Multi-Speaker Mode

-   **Multi-Speaker Mode**: `ENABLED` / `DISABLED`
    -   This is the master switch for the feature. When disabled, the entire text will be generated with a single, default voice, and speaker labels will be ignored.

### Generation Methods

-   **Generation Method**: `auto` / `random` / `dataset` / `audio`
    -   Determines how new voices are created. See the detailed section below for more information.

### Voice Customization

These options fine-tune the voice generation process:

-   **Default Gender**: `neutral` / `male` / `female`
    -   When using the `dataset` method, this filters the pool of available voices to better match your desired output. It is a heuristic and not always perfect.
-   **Random Seed**: (Any integer)
    -   Setting a seed makes voice generation deterministic. With a seed, the same speaker name will **always** get the same voice, even across different LlamaNote sessions or machines. Leave it blank for random, non-deterministic voices.
-   **Distribution**: `gaussian` / `uniform`
    -   For the `random` method. `gaussian` (default) tends to produce more centered, typical voices, while `uniform` can create more varied and sometimes unusual voices.

## Embedding Generation Methods in Detail

### Random (Default)

-   **Method**: `random`
-   **How it works**: Creates a new embedding by drawing random numbers from a statistical distribution (`gaussian` or `uniform`).
-   **Pros**: Infinitely varied, very fast, no external dependencies.
-   **Cons**: Voices are synthetic and may occasionally sound less natural than real human voices.
-   **Best for**: Quickly creating distinct voices for fictional characters or when naturalness is not the top priority.

### Dataset

-   **Method**: `dataset`
-   **How it works**: Samples a real speaker embedding from a pre-computed dataset. The default dataset is `Matthijs/cmu-arctic-xvectors`, which contains over 7,000 unique speakers.
-   **Pros**: High-quality, natural-sounding human voices.
-   **Cons**: Requires downloading the dataset (~500MB), limited to the voices in the dataset.
-   **Best for**: Generating high-quality podcasts, audiobooks, or any content where voice naturalness is important.

### Audio (Voice Cloning)

-   **Method**: `audio`
-   **Status**: **Coming in a future update.**
-   **How it will work**: This method will allow you to provide your own audio file (e.g., a `.wav` of someone speaking). LlamaNote will analyze the audio and extract a speaker embedding from it, effectively cloning the voice.
-   **Pros**: Clone any voice you have an audio sample of.
-   **Cons**: Requires high-quality, clean audio (at least 5-10 seconds of speech is recommended).

## Caching and Persistence

To ensure consistency, LlamaNote automatically caches generated speaker embeddings.

-   **Location**: `cache/speakers/`
-   **Files**:
    -   `speaker_mappings.json`: A map of speaker names to their assigned embedding metadata.
    -   `[Speaker Name].npy`: The actual embedding vector for each speaker.

**How it works:**

-   When you run audio generation, the `SpeakerEmbeddingManager` first checks this cache.
-   If an embedding for a speaker (e.g., "Host") exists, it's loaded directly.
-   If not, a new one is generated and saved to the cache for future use.

**To get new voices for existing speakers, you must clear the cache.** You can do this from the **Speaker Embeddings** menu.

## Troubleshooting

-   **Problem**: All speakers have the same voice.
    -   **Solution**: Ensure **Multi-Speaker Mode** is `ENABLED` in the Speaker Embeddings menu. Also, verify your text is formatted correctly with `**[Speaker NAME]:**` labels.

-   **Problem**: The voices are different each time I run the program.
    -   **Solution**: Set a **Random Seed** in the Speaker Embeddings menu. This makes the voice assignment deterministic.

-   **Problem**: I want to change the voice for a specific speaker.
    -   **Solution**: Go to the Speaker Embeddings menu and select **Clear Speaker Cache**. The next time you run generation, new voices will be created for all speakers.