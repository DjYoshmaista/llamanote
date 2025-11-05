"""
Speech-to-text transcript generation using Whisper models.

Generates transcripts from audio files using HuggingFace Whisper models,
with configurable model selection and quality settings.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from difflib import SequenceMatcher
import soundfile as sf

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)

from ...utils.logger import ContextLogger
from ...models.model_manager import ModelManager


class TranscriptGenerator:
    """
    Generates transcripts from audio using Whisper speech-to-text models.

    Supports configurable model selection with sensible defaults (tiny.en).
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-tiny.en",
        device: Optional[str] = None,
        logger: Optional[ContextLogger] = None,
        model_manager: Optional[ModelManager] = None
    ):
        """
        Initialize transcript generator.

        Args:
            model_name: HuggingFace Whisper model name (default: tiny.en for speed)
            device: Device to use ('cuda' or 'cpu'), auto-detected if None
            logger: Optional logger instance
            model_manager: Optional model manager for downloading/caching
        """
        self.model_name = model_name
        self.logger = logger or ContextLogger("TranscriptGenerator")
        self.model_manager = model_manager

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.logger.info(f"Initializing Whisper transcription with {model_name} on {self.device}")

        self.pipe = None
        self.processor = None
        self.model = None

    def _load_model(self):
        """Load Whisper model and processor."""
        if self.pipe is not None:
            return  # Already loaded

        try:
            self.logger.info(f"Loading Whisper model: {self.model_name}")

            # Use pipeline for simplicity and efficiency
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            self.logger.info("Whisper model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_audio_file(
        self,
        audio_path: Path,
        language: str = "en",
        return_timestamps: bool = False
    ) -> Optional[str]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (default: 'en')
            return_timestamps: Whether to include timestamps

        Returns:
            Transcribed text or None on failure
        """
        if not audio_path.exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            return None

        try:
            self._load_model()

            self.logger.debug(f"Transcribing audio: {audio_path}")

            # Transcribe using pipeline
            result = self.pipe(
                str(audio_path),
                return_timestamps=return_timestamps,
                generate_kwargs={"language": language}
            )

            # Extract text
            if isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)

            self.logger.debug(f"Transcription complete: {len(text)} characters")
            return text.strip()

        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            return None

    def transcribe_audio_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: str = "en"
    ) -> Optional[str]:
        """
        Transcribe audio from numpy array.

        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of audio
            language: Language code

        Returns:
            Transcribed text or None on failure
        """
        try:
            self._load_model()

            self.logger.debug(f"Transcribing audio array: {audio_array.shape}")

            # Pipeline expects 16kHz audio for Whisper
            target_sr = 16000

            if sample_rate != target_sr:
                # Resample audio
                audio_array = self._resample_audio(audio_array, sample_rate, target_sr)

            # Transcribe
            result = self.pipe(
                {"array": audio_array, "sampling_rate": target_sr},
                generate_kwargs={"language": language}
            )

            # Extract text
            if isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)

            return text.strip()

        except Exception as e:
            self.logger.error(f"Failed to transcribe audio array: {e}")
            return None

    def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            self.logger.warning("librosa not available, using scipy for resampling")
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples)

    def compare_texts(
        self,
        original_text: str,
        transcribed_text: str
    ) -> Dict[str, Any]:
        """
        Compare original text with transcribed text.

        Args:
            original_text: Original text used for generation
            transcribed_text: Transcribed text from audio

        Returns:
            Comparison results dictionary
        """
        # Normalize texts for comparison
        orig_normalized = original_text.lower().strip()
        trans_normalized = transcribed_text.lower().strip()

        # Calculate similarity using SequenceMatcher
        similarity = SequenceMatcher(None, orig_normalized, trans_normalized).ratio()

        # Find differences
        differences = []

        if orig_normalized != trans_normalized:
            # Character-level differences
            matcher = SequenceMatcher(None, orig_normalized, trans_normalized)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != 'equal':
                    differences.append({
                        "type": tag,
                        "original": original_text[i1:i2],
                        "transcribed": transcribed_text[j1:j2],
                        "position": {"original": (i1, i2), "transcribed": (j1, j2)}
                    })

        comparison = {
            "similarity_score": round(similarity, 4),
            "original_length": len(original_text),
            "transcribed_length": len(transcribed_text),
            "length_difference": len(transcribed_text) - len(original_text),
            "num_differences": len(differences),
            "differences": differences[:10]  # Limit to first 10 for report size
        }

        return comparison

    def generate_transcript_with_comparison(
        self,
        audio_path: Path,
        original_text: str,
        output_transcript_path: Path,
        output_original_path: Path,
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate transcript and compare with original text.

        Args:
            audio_path: Path to audio file
            original_text: Original text used for generation
            output_transcript_path: Path to save transcript
            output_original_path: Path to save original text
            language: Language code

        Returns:
            Comparison results or None on failure
        """
        # Generate transcript
        transcript = self.transcribe_audio_file(audio_path, language=language)

        if transcript is None:
            return None

        # Save transcript
        try:
            output_transcript_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_transcript_path, 'w', encoding='utf-8') as f:
                f.write(f"# Transcript (Generated via Speech-to-Text)\n\n")
                f.write(f"**Audio File:** {audio_path.name}\n\n")
                f.write(f"**Model:** {self.model_name}\n\n")
                f.write(f"---\n\n")
                f.write(transcript)

            self.logger.debug(f"Saved transcript: {output_transcript_path}")

        except Exception as e:
            self.logger.error(f"Failed to save transcript: {e}")
            return None

        # Save original text
        try:
            output_original_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_original_path, 'w', encoding='utf-8') as f:
                f.write(f"# Original Text (Used for Audio Generation)\n\n")
                f.write(f"---\n\n")
                f.write(original_text)

            self.logger.debug(f"Saved original text: {output_original_path}")

        except Exception as e:
            self.logger.error(f"Failed to save original text: {e}")

        # Compare texts
        comparison = self.compare_texts(original_text, transcript)

        self.logger.info(
            f"Transcript similarity: {comparison['similarity_score']:.2%} "
            f"({comparison['num_differences']} differences)"
        )

        return comparison

    def unload_model(self):
        """Unload model from memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Unloaded Whisper model")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_model()
        except:
            pass
