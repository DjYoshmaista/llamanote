# llamanote/processing/audio_processor.py
"""
Audio Post-Processing Module
Handles operations on generated audio files, like adding background music,
silence, and applying effects.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import warnings

from ..utils.logger import get_logger_conf
from ..utils.decorators import log_execution_time
from ..core.errors import FileProcessingError

# Suppress warnings from audio libraries if they get too noisy
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

logger = get_logger_conf(__name__)

# --- Lazy Loading for heavy dependencies ---
_soundfile = None
_librosa = None

def get_sf():
    """Lazy load soundfile."""
    global _soundfile
    if _soundfile is None:
        try:
            import soundfile as sf
            _soundfile = sf
        except ImportError:
            logger.error("Soundfile library not found. Please install with 'pip install soundfile'.")
            raise
    return _soundfile

def get_librosa():
    """Lazy load librosa."""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
        except ImportError:
            logger.error("Librosa library not found. Please install with 'pip install librosa'.")
            raise
    return _librosa


class AudioPostProcessor:
    """A class for applying post-processing effects to audio files."""

    @staticmethod
    def _load_audio(path: Path, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Loads an audio file using librosa, handling potential errors."""
        librosa = get_librosa()
        try:
            audio, sample_rate = librosa.load(path, sr=sr)
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio file {path}: {e}", exc_info=True)
            raise FileProcessingError(f"Failed to load audio: {e}", str(path))

    @staticmethod
    def _save_audio(path: Path, audio: np.ndarray, sr: int, format: str = "wav", subtype: str = 'PCM_16'):
        """Saves an audio array to a file using soundfile."""
        sf = get_sf()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, sr, subtype=subtype, format=format.upper())
            logger.info(f"Saved post-processed audio to {path}")
        except Exception as e:
            logger.error(f"Failed to save audio file {path}: {e}", exc_info=True)
            raise FileProcessingError(f"Failed to save audio: {e}", str(path))

    @staticmethod
    def _normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalizes audio to a target peak amplitude."""
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        max_val = np.abs(audio).max()
        if max_val > 1e-6: # Avoid division by zero
            return (audio / max_val) * target_peak
        return audio # Return silent audio as-is

    @staticmethod
    @log_execution_time(logger_name=__name__)
    def add_background_music(speech_path: Path,
                            music_path: Path,
                            output_path: Path,
                            music_volume: float = 0.2) -> Optional[Path]:
        """
        Mixes speech with background music.

        Args:
            speech_path: Path to the main speech audio file.
            music_path: Path to the background music file.
            output_path: Path to save the mixed audio file.
            music_volume: Factor to scale music volume (0.0 to 1.0).

        Returns:
            Path to the saved file, or None on failure.
        """
        librosa = get_librosa()
        try:
            speech, sr = AudioPostProcessor._load_audio(speech_path, sr=None)
            music, sr_music = AudioPostProcessor._load_audio(music_path, sr=sr) # Resample music to match speech SR

            # Ensure music is at least as long as speech (loop if necessary)
            if len(music) < len(speech):
                repeats = int(np.ceil(len(speech) / len(music)))
                music = np.tile(music, repeats)

            # Trim music to match speech length
            music_trimmed = music[:len(speech)]

            # Normalize speech first (optional, but good practice)
            speech_norm = AudioPostProcessor._normalize(speech, 0.70) # Normalize speech to 70%
            music_norm = AudioPostProcessor._normalize(music_trimmed, 0.70) # Normalize music to 70%

            # Mix audio with specified volume
            mixed_audio = speech_norm + (music_norm * music_volume)

            # Re-normalize the final mix to prevent clipping
            mixed_audio_final = AudioPostProcessor._normalize(mixed_audio, 0.95)
            
            # Save the mixed audio
            AudioPostProcessor._save_audio(output_path, mixed_audio_final, sr, format=output_path.suffix[1:])
            return output_path
        except Exception as e:
            logger.error(f"Failed to add background music: {e}", exc_info=True)
            return None

    @staticmethod
    @log_execution_time(logger_name=__name__)
    def add_silence(audio_path: Path,
                   output_path: Optional[Path] = None,
                   start_seconds: float = 0.5,
                   end_seconds: float = 1.0) -> Optional[Path]:
        """
        Adds silence to the beginning and/or end of an audio file.

        Args:
            audio_path: Path to the source audio file.
            output_path: Path to save the new file. If None, overwrites original.
            start_seconds: Duration of silence to add at the beginning.
            end_seconds: Duration of silence to add at the end.

        Returns:
            Path to the saved file, or None on failure.
        """
        try:
            audio, sr = AudioPostProcessor._load_audio(audio_path, sr=None)

            start_samples = int(start_seconds * sr)
            end_samples = int(end_seconds * sr)

            # Create silence arrays
            start_padding = np.zeros(start_samples, dtype=audio.dtype)
            end_padding = np.zeros(end_samples, dtype=audio.dtype)

            # Concatenate
            padded_audio = np.concatenate([start_padding, audio, end_padding])

            save_path = output_path or audio_path # Overwrite if no output path given
            AudioPostProcessor._save_audio(save_path, padded_audio, sr, format=save_path.suffix[1:])
            
            logger.info(f"Added silence to {audio_path.name}. Output: {save_path.name}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to add silence to {audio_path.name}: {e}", exc_info=True)
            return None

    @staticmethod
    @log_execution_time(logger_name=__name__)
    def apply_effects(audio_path: Path,
                     output_path: Optional[Path] = None,
                     reverb_amount: float = 0.0,
                     echo_delay_s: float = 0.0,
                     echo_decay: float = 0.5) -> Optional[Path]:
        """
        Applies simple audio effects.

        Args:
            audio_path: Path to the source audio file.
            output_path: Path to save the new file. If None, overwrites original.
            reverb_amount: Amount of reverb (0.0 to 1.0). Requires scipy.
            echo_delay_s: Echo delay in seconds.
            echo_decay: Echo feedback decay (0.0 to 1.0).

        Returns:
            Path to the saved file, or None on failure.
        """
        try:
            audio, sr = AudioPostProcessor._load_audio(audio_path, sr=None)
            processed_audio = audio.copy().astype(np.float32) # Work with float

            # Simple Reverb
            if reverb_amount > 0.0:
                logger.debug(f"Applying reverb (amount: {reverb_amount:.2f})")
                try:
                     from scipy.signal import convolve
                     decay_time = reverb_amount * 2 # Heuristic
                     ir_length = int(decay_time * sr)
                     if ir_length > 0:
                          t = np.linspace(0, decay_time, ir_length)
                          impulse_response = np.exp(-5 * t / decay_time) * np.random.randn(ir_length)
                          impulse_response /= np.sum(np.abs(impulse_response)) # Normalize
                          processed_audio = convolve(processed_audio, impulse_response, mode='same')
                except ImportError:
                     logger.warning("scipy not installed. Skipping reverb effect.")
                except Exception as reverb_err:
                     logger.error(f"Error applying reverb: {reverb_err}")

            # Simple Echo
            if echo_delay_s > 0.0 and 0.0 < echo_decay < 1.0:
                logger.debug(f"Applying echo (delay: {echo_delay_s:.2f}s, decay: {echo_decay:.2f})")
                try:
                     librosa = get_librosa()
                     delay_samples = int(echo_delay_s * sr)
                     if delay_samples > 0 and delay_samples < len(processed_audio):
                          echo_component = np.zeros_like(processed_audio)
                          # Add delayed signal with decay
                          echo_component[delay_samples:] = processed_audio[:-delay_samples] * echo_decay
                          # Simple single echo
                          processed_audio = processed_audio + echo_component
                     else:
                          logger.warning("Echo delay is too long or zero, skipping echo effect.")
                except Exception as echo_err:
                     logger.error(f"Error applying echo: {echo_err}")

            # Normalize final audio
            processed_audio = AudioPostProcessor._normalize(processed_audio, 0.95)

            save_path = output_path or audio_path
            AudioPostProcessor._save_audio(save_path, processed_audio, sr, format=save_path.suffix[1:])
            
            logger.info(f"Applied effects to {audio_path.name}. Output: {save_path.name}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to apply audio effects: {e}", exc_info=True)
            return None
