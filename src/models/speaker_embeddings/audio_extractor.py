# src/models/speaker_embeddings/audio_extractor.py
"""
Audio-Based Speaker Embedding Extraction

Extracts speaker embeddings from audio files using pre-trained models.
Supports voice cloning by analyzing audio samples and creating embeddings
that can be used with TTS models.

Key features:
- SpeechBrain integration for x-vector extraction
- Audio preprocessing (resampling, VAD, normalization)
- Quality assessment and validation
- Multi-file averaging for better quality
- Support for various audio formats

Pros:
- High-quality embeddings from real voices
- Voice cloning capability
- Preserves speaker characteristics
- Better than random or dataset for specific voices

Cons:
- Requires audio files (3+ seconds recommended)
- Processing time (1-5 seconds per file)
- Quality depends on input audio
- Requires additional dependencies (SpeechBrain, librosa)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import warnings

from .base import SpeakerEmbeddingGenerator, EmbeddingConfig, EmbeddingMethod
from ...utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

# Check for optional dependencies
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available - audio extraction will not work")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - audio extraction will not work")

try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("SpeechBrain not available - audio extraction will not work")


class AudioQualityAssessment:
    """Results from audio quality assessment."""

    def __init__(
        self,
        is_valid: bool,
        duration: float,
        sample_rate: int,
        snr_db: Optional[float] = None,
        has_clipping: bool = False,
        is_silent: bool = False,
        warnings: List[str] = None,
        recommendations: List[str] = None
    ):
        self.is_valid = is_valid
        self.duration = duration
        self.sample_rate = sample_rate
        self.snr_db = snr_db
        self.has_clipping = has_clipping
        self.is_silent = is_silent
        self.warnings = warnings or []
        self.recommendations = recommendations or []

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"AudioQualityAssessment({status}, "
            f"duration={self.duration:.1f}s, "
            f"sr={self.sample_rate}Hz, "
            f"warnings={len(self.warnings)})"
        )


class AudioEmbeddingExtractor(SpeakerEmbeddingGenerator):
    """
    Extract speaker embeddings from audio files.

    Uses pre-trained speaker recognition models (SpeechBrain) to extract
    embeddings from audio recordings. These embeddings capture the speaker's
    voice characteristics and can be used for voice cloning.

    Example:
        extractor = AudioEmbeddingExtractor()

        # Extract from single file
        embedding = extractor.generate(audio_path="speaker_sample.wav")

        # Extract from multiple files and average
        embedding = extractor.extract_from_multiple(
            audio_paths=["sample1.wav", "sample2.wav", "sample3.wav"],
            speaker_name="Custom_Voice"
        )

        # With quality assessment
        embedding, quality = extractor.extract_with_quality(
            audio_path="speaker_sample.wav"
        )
    """

    # Supported audio formats
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']

    # Model configurations
    SPEECHBRAIN_MODELS = {
        'xvector': 'speechbrain/spkrec-xvect-voxceleb',
        'ecapa': 'speechbrain/spkrec-ecapa-voxceleb',
    }

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        model_type: str = 'xvector',
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize audio embedding extractor.

        Args:
            config: EmbeddingConfig with audio parameters
            model_type: Model to use ('xvector' or 'ecapa')
            cache_dir: Directory for model cache
        """
        super().__init__(config)

        if not all([SOUNDFILE_AVAILABLE, LIBROSA_AVAILABLE, SPEECHBRAIN_AVAILABLE]):
            raise RuntimeError(
                "Audio extraction requires: soundfile, librosa, speechbrain. "
                "Install with: pip install soundfile librosa speechbrain"
            )

        self.model_type = model_type
        self.cache_dir = cache_dir or Path("cache/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded model
        self._model: Optional[EncoderClassifier] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"AudioEmbeddingExtractor initialized: model={model_type}, device={self._device}")

    def _init_model(self):
        """Lazy initialization of SpeechBrain model."""
        if self._model is not None:
            return

        try:
            model_id = self.SPEECHBRAIN_MODELS.get(self.model_type)
            if model_id is None:
                raise ValueError(f"Unknown model type: {self.model_type}")

            logger.info(f"Loading SpeechBrain model: {model_id}")

            self._model = EncoderClassifier.from_hparams(
                source=model_id,
                savedir=str(self.cache_dir / self.model_type),
                run_opts={"device": self._device}
            )

            logger.info(f"Model loaded successfully on {self._device}")

        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize audio extraction model: {e}") from e

    def generate(
        self,
        speaker_name: Optional[str] = None,
        audio_path: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate embedding from audio file.

        Args:
            speaker_name: Optional speaker identifier (for logging)
            audio_path: Path to audio file (required for audio method)
            **kwargs: Additional parameters:
                - preprocess: Whether to preprocess audio (default True)
                - validate: Whether to validate audio quality (default True)

        Returns:
            np.ndarray: Speaker embedding extracted from audio

        Raises:
            ValueError: If audio_path not provided or file doesn't exist
            RuntimeError: If extraction fails
        """
        # Get audio path from kwargs or config
        if audio_path is None:
            audio_path = kwargs.get('audio_path', self.config.audio_path)

        if audio_path is None:
            raise ValueError("audio_path required for audio-based extraction")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        # Options
        preprocess = kwargs.get('preprocess', True)
        validate = kwargs.get('validate', True)

        logger.info(f"Extracting embedding from audio: {audio_path.name}")

        try:
            # Load and preprocess audio
            waveform, sr = self._load_audio(audio_path)

            if validate:
                quality = self._assess_quality(waveform, sr, audio_path)
                if not quality.is_valid:
                    logger.warning(f"Audio quality issues: {quality.warnings}")
                    for warning in quality.warnings:
                        logger.warning(f"  - {warning}")
                    for rec in quality.recommendations:
                        logger.info(f"  Recommendation: {rec}")

            if preprocess:
                waveform, sr = self._preprocess_audio(waveform, sr)

            # Extract embedding
            embedding = self._extract_embedding(waveform, sr)

            # Normalize
            if self.config.normalize:
                embedding = self.normalize_embedding(embedding)

            logger.info(f"Successfully extracted embedding from {audio_path.name}")
            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to extract embedding from {audio_path}: {e}", exc_info=True)
            raise RuntimeError(f"Audio extraction failed: {e}") from e

    def extract_with_quality(
        self,
        audio_path: str,
        speaker_name: Optional[str] = None
    ) -> Tuple[np.ndarray, AudioQualityAssessment]:
        """
        Extract embedding and return quality assessment.

        Args:
            audio_path: Path to audio file
            speaker_name: Optional speaker identifier

        Returns:
            Tuple of (embedding, quality_assessment)
        """
        audio_path = Path(audio_path)

        # Load audio
        waveform, sr = self._load_audio(audio_path)

        # Assess quality
        quality = self._assess_quality(waveform, sr, audio_path)

        # Extract embedding (with preprocessing)
        waveform_proc, sr_proc = self._preprocess_audio(waveform, sr)
        embedding = self._extract_embedding(waveform_proc, sr_proc)

        if self.config.normalize:
            embedding = self.normalize_embedding(embedding)

        return embedding.astype(np.float32), quality

    def extract_from_multiple(
        self,
        audio_paths: List[str],
        speaker_name: Optional[str] = None,
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Extract and average embeddings from multiple audio files.

        Using multiple audio samples of the same speaker improves embedding
        quality and robustness.

        Args:
            audio_paths: List of paths to audio files
            speaker_name: Optional speaker identifier
            weights: Optional weights for averaging (default: equal weights)

        Returns:
            np.ndarray: Averaged embedding
        """
        if not audio_paths:
            raise ValueError("At least one audio file required")

        logger.info(f"Extracting embeddings from {len(audio_paths)} files")

        embeddings = []
        valid_weights = []

        for i, audio_path in enumerate(audio_paths):
            try:
                embedding = self.generate(audio_path=audio_path, validate=True, preprocess=True)
                embeddings.append(embedding)

                # Use weight if provided, otherwise equal weight
                weight = weights[i] if weights and i < len(weights) else 1.0
                valid_weights.append(weight)

                logger.debug(f"  Extracted from {Path(audio_path).name}")

            except Exception as e:
                logger.warning(f"  Failed to extract from {Path(audio_path).name}: {e}")
                continue

        if not embeddings:
            raise RuntimeError("Failed to extract embeddings from any file")

        # Average embeddings
        averaged = self._average_embeddings(embeddings, valid_weights)

        logger.info(f"Averaged {len(embeddings)}/{len(audio_paths)} embeddings successfully")
        return averaged

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            # Try soundfile first (fastest)
            waveform, sr = sf.read(str(audio_path), dtype='float32')

            # Convert stereo to mono
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)

            logger.debug(f"Loaded audio: {audio_path.name} ({waveform.shape[0]/sr:.1f}s, {sr}Hz)")
            return waveform, sr

        except Exception as e:
            logger.debug(f"soundfile failed, trying librosa: {e}")

            try:
                # Fallback to librosa
                waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)
                logger.debug(f"Loaded audio with librosa: {audio_path.name}")
                return waveform, sr

            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file {audio_path}: {e2}") from e2

    def _preprocess_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio for embedding extraction.

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate

        Returns:
            Tuple of (preprocessed_waveform, target_sample_rate)
        """
        # Target sample rate for SpeechBrain models
        target_sr = 16000

        # Resample if needed
        if sample_rate != target_sr:
            logger.debug(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=target_sr
            )
            sample_rate = target_sr

        # Normalize amplitude
        if np.abs(waveform).max() > 0:
            waveform = waveform / np.abs(waveform).max() * 0.95

        # Trim silence
        waveform, _ = librosa.effects.trim(
            waveform,
            top_db=20,
            frame_length=2048,
            hop_length=512
        )

        logger.debug(f"Preprocessed audio: {waveform.shape[0]/sample_rate:.1f}s")
        return waveform, sample_rate

    def _extract_embedding(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract embedding using SpeechBrain model.

        Args:
            waveform: Preprocessed audio waveform
            sample_rate: Sample rate

        Returns:
            np.ndarray: Extracted embedding
        """
        # Initialize model if needed
        self._init_model()

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(waveform).float()

            # Add batch dimension if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Move to device
            audio_tensor = audio_tensor.to(self._device)

            # Extract embedding
            with torch.no_grad():
                embedding = self._model.encode_batch(audio_tensor)

            # Convert to numpy
            embedding_np = embedding.squeeze().cpu().numpy()

            logger.debug(f"Extracted embedding shape: {embedding_np.shape}")
            return embedding_np

        except Exception as e:
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

    def _assess_quality(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        audio_path: Path
    ) -> AudioQualityAssessment:
        """
        Assess audio quality for embedding extraction.

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            audio_path: Path to audio file

        Returns:
            AudioQualityAssessment: Quality assessment results
        """
        warnings_list = []
        recommendations = []
        is_valid = True

        # Check duration
        duration = len(waveform) / sample_rate
        if duration < self.config.audio_duration_min:
            warnings_list.append(
                f"Audio too short ({duration:.1f}s < {self.config.audio_duration_min}s minimum)"
            )
            recommendations.append(f"Use audio file at least {self.config.audio_duration_min}s long")
            is_valid = False

        # Check for silence
        energy = np.abs(waveform).mean()
        if energy < 0.01:
            warnings_list.append("Audio appears to be silent or very quiet")
            recommendations.append("Check audio file, ensure it contains speech")
            is_valid = False

        # Check for clipping
        max_amplitude = np.abs(waveform).max()
        has_clipping = max_amplitude >= 0.99
        if has_clipping:
            warnings_list.append("Audio has clipping (distortion)")
            recommendations.append("Re-record with lower gain to avoid clipping")

        # Estimate SNR (rough approximation)
        snr_db = None
        try:
            # Simple energy-based SNR estimate
            signal_power = np.mean(waveform ** 2)
            if signal_power > 0:
                snr_db = 10 * np.log10(signal_power / (np.std(waveform) ** 2 + 1e-10))

                if snr_db < 10:
                    warnings_list.append(f"Low SNR ({snr_db:.1f} dB)")
                    recommendations.append("Record in quieter environment")

        except Exception:
            pass

        # Check sample rate
        if sample_rate < 8000:
            warnings_list.append(f"Low sample rate ({sample_rate}Hz)")
            recommendations.append("Use audio with at least 16kHz sample rate")

        return AudioQualityAssessment(
            is_valid=is_valid,
            duration=duration,
            sample_rate=sample_rate,
            snr_db=snr_db,
            has_clipping=has_clipping,
            is_silent=energy < 0.01,
            warnings=warnings_list,
            recommendations=recommendations
        )

    def _average_embeddings(
        self,
        embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Average multiple embeddings with optional weights.

        Args:
            embeddings: List of embedding arrays
            weights: Optional weights for averaging

        Returns:
            np.ndarray: Averaged and normalized embedding
        """
        if not embeddings:
            raise ValueError("No embeddings to average")

        if len(embeddings) == 1:
            return embeddings[0]

        # Stack embeddings
        embeddings_array = np.stack(embeddings, axis=0)

        # Apply weights if provided
        if weights is not None:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            averaged = np.average(embeddings_array, axis=0, weights=weights)
        else:
            averaged = np.mean(embeddings_array, axis=0)

        # Re-normalize
        if self.config.normalize:
            averaged = self.normalize_embedding(averaged)

        return averaged.astype(np.float32)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return self.SUPPORTED_FORMATS.copy()

    def validate_audio_file(self, audio_path: str) -> Tuple[bool, str]:
        """
        Validate audio file without extracting embedding.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (is_valid, message)
        """
        audio_path = Path(audio_path)

        # Check exists
        if not audio_path.exists():
            return False, f"File not found: {audio_path}"

        # Check format
        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {audio_path.suffix}"

        try:
            # Try to load
            waveform, sr = self._load_audio(audio_path)

            # Assess quality
            quality = self._assess_quality(waveform, sr, audio_path)

            if quality.is_valid:
                return True, f"Valid audio file ({quality.duration:.1f}s, {quality.sample_rate}Hz)"
            else:
                return False, "; ".join(quality.warnings)

        except Exception as e:
            return False, f"Failed to load audio: {e}"

    def get_info(self) -> Dict[str, Any]:
        """Get extractor information."""
        info = super().get_info()
        info.update({
            "model_type": self.model_type,
            "model_loaded": self._model is not None,
            "device": self._device,
            "supported_formats": self.SUPPORTED_FORMATS,
            "cache_dir": str(self.cache_dir),
        })
        return info
