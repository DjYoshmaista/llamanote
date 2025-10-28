"""
Audio Generator Module - Refactored with Backend Abstraction
Text-to-speech generation using local HuggingFace models or cloud providers
"""
import gc
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import soundfile as sf
import librosa
from transformers import pipeline, AutoProcessor, AutoModel # Keep for local models
import warnings
import time
import abc

# --- LlamaNote Modules ---
from loggerConf import get_logger_conf, log_execution_time, ConsoleOutput
from huggingface_search import ModelDownloader # Keep for local models
from config_manager import ConfigManager # Not directly needed here, but used by callers
# Imports for local model types (conditionally imported in LocalAudioBackend)

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger_conf(__name__)

# --- Configuration Dataclasses ---

@dataclass
class AudioConfig:
    """Configuration for audio generation (common settings)."""
    # model_id is now provider-specific, handled by AppState or PipelineConfig
    speaker_embedding: Optional[str] = None # Relevant for some local models like SpeechT5
    sample_rate: int = 44100 # Target sample rate, backends might resample
    output_format: str = "wav" # Preferred output format (wav, mp3, flac)
    chunk_size: int = 3000  # Characters per text chunk for local models if needed
    speed: float = 1.0 # Playback speed factor (requires post-processing)
    pitch_shift: int = 0  # Semitones for pitch shift (requires post-processing)
    volume_normalize: bool = True # Whether to normalize volume (post-processing)
    device: str = "auto"  # auto, cpu, cuda (mainly for local models)
    use_half_precision: bool = True # For local models
    # Cloud specific settings might go here or in backend-specific configs if complex
    cloud_voice: str = "alloy" # Example: default voice for cloud providers like OpenAI

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Use dataclasses.asdict for simplicity
        from dataclasses import asdict
        return asdict(self)


@dataclass
class AudioResult:
    """Result from audio generation."""
    audio_path: Path
    duration_seconds: float
    sample_rate: int
    num_samples: int
    model_used: str # Identifier including provider, e.g., "local:microsoft/speecht5_tts", "openai:tts-1"
    processing_time: float
    chunks_processed: int # How many text chunks were sent to the TTS engine

    @property
    def duration_formatted(self) -> str:
        """Get formatted duration string."""
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes}:{seconds:02d}"

# --- Abstract Audio Backend ---

class AudioBackend(abc.ABC):
    """Abstract base class for audio generation backends."""

    def __init__(self, config: AudioConfig, model_specifier: str):
        self.config = config
        self.model_specifier = model_specifier # e.g., HF ID or cloud model name
        self.logger = get_logger_conf(f"{self.__class__.__name__}") # Logger for subclass

    @abc.abstractmethod
    def load_model(self, **kwargs) -> bool:
        """Load or initialize the backend."""
        pass

    @abc.abstractmethod
    @log_execution_time()
    def generate_audio(self,
                      text: str,
                      output_path: Optional[Path] = None,
                      chunk_text: bool = True, # Backend decides if needed
                      **kwargs) -> Optional[AudioResult]:
        """Generate audio from text."""
        pass

    @abc.abstractmethod
    def unload_model(self):
        """Unload resources."""
        pass

    @property
    @abc.abstractmethod
    def provider_identifier(self) -> str:
        """Return provider name (e.g., 'local', 'openai')."""
        pass

    @property
    def model_identifier(self) -> str:
        """Return full model identifier including provider."""
        return f"{self.provider_identifier}:{self.model_specifier}"

    # --- Common Helper Methods (Can be overridden) ---

    def _setup_device(self) -> str:
        """Setup compute device (mainly for local models)."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks based on config size (simple sentence split)."""
        chunks = []
        # Attempt to split by sentences first, then fall back if needed
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text) # Basic sentence split
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue

            # Check if adding the sentence exceeds chunk size
            if len(current_chunk) > 0 and len(current_chunk) + len(sentence) + 1 > self.config.chunk_size:
                 # If current chunk is not empty, add it
                 chunks.append(current_chunk)
                 current_chunk = sentence # Start new chunk with current sentence
            elif len(sentence) > self.config.chunk_size:
                 # If a single sentence is too long, split it crudely (or handle error)
                 # Add any existing chunk first
                 if current_chunk:
                      chunks.append(current_chunk)
                 # Split the long sentence
                 parts = [sentence[i:i+self.config.chunk_size] for i in range(0, len(sentence), self.config.chunk_size)]
                 chunks.extend(parts)
                 current_chunk = "" # Reset chunk
            else:
                 # Add sentence to current chunk
                 if current_chunk:
                     current_chunk += " " + sentence
                 else:
                     current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        # Filter out empty chunks that might result from splitting
        return [c for c in chunks if c]


    def _combine_audio(self, audio_arrays: List[np.ndarray]) -> Optional[np.ndarray]:
        """Combine multiple audio arrays, adding slight silence."""
        if not audio_arrays: return None
        if len(audio_arrays) == 1: return audio_arrays[0]

        # Ensure all arrays are numpy arrays and handle potential None values
        valid_arrays = [arr for arr in audio_arrays if isinstance(arr, np.ndarray) and arr.size > 0]
        if not valid_arrays: return None
        if len(valid_arrays) == 1: return valid_arrays[0]

        # Add small silence between chunks
        silence_samples = int(0.2 * self.config.sample_rate)  # 200ms
        silence = np.zeros(silence_samples, dtype=valid_arrays[0].dtype) # Match dtype

        combined = []
        for i, audio in enumerate(valid_arrays):
            combined.append(audio)
            if i < len(valid_arrays) - 1:
                combined.append(silence)

        try:
            return np.concatenate(combined)
        except ValueError as e:
             self.logger.error(f"Error concatenating audio chunks: {e}")
             # Log shapes for debugging
             for i, arr in enumerate(valid_arrays):
                 self.logger.debug(f"Chunk {i} shape: {arr.shape}, dtype: {arr.dtype}")
             return None # Or handle more gracefully

    def _post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Post-process audio (speed, pitch, normalization)."""
        if not isinstance(audio, np.ndarray) or audio.size == 0:
             self.logger.warning("Invalid audio array passed to post-processing.")
             return np.array([]) # Return empty array

        processed_audio = audio.copy() # Work on a copy

        try:
            # Speed adjustment
            if self.config.speed != 1.0:
                 # Librosa expects float audio
                 processed_audio = processed_audio.astype(np.float32)
                 processed_audio = librosa.effects.time_stretch(processed_audio, rate=self.config.speed)

            # Pitch shift
            if self.config.pitch_shift != 0:
                 # Librosa expects float audio
                 processed_audio = processed_audio.astype(np.float32)
                 processed_audio = librosa.effects.pitch_shift(
                     y=processed_audio, # Use y= argument
                     sr=self.config.sample_rate,
                     n_steps=self.config.pitch_shift
                 )

            # Volume normalization
            if self.config.volume_normalize:
                 # Operate on float audio for normalization
                 processed_audio = processed_audio.astype(np.float32)
                 max_val = np.abs(processed_audio).max()
                 if max_val > 1e-6: # Avoid division by zero or near-zero
                     processed_audio = processed_audio / max_val * 0.95  # Normalize to 95%
                 else:
                     self.logger.warning("Audio max value is near zero, skipping normalization.")

            # Ensure final audio is in a suitable format for saving (e.g., float32 for wav/flac, maybe int16?)
            # Soundfile handles float32 well for WAV.
            processed_audio = processed_audio.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error during audio post-processing: {e}. Returning original audio.")
            return audio # Return original if processing fails

        return processed_audio


    def _save_audio(self, audio: np.ndarray, output_path: Path) -> Optional[Path]:
        """Save audio to file using soundfile."""
        if not isinstance(audio, np.ndarray) or audio.size == 0:
             self.logger.error("Cannot save empty or invalid audio array.")
             return None

        output_path = Path(output_path)

        # Ensure extension matches format, default to wav
        target_format = self.config.output_format.lower()
        if target_format not in ["wav", "mp3", "flac"]:
            self.logger.warning(f"Unsupported output format '{target_format}', defaulting to 'wav'.")
            target_format = "wav"

        output_path = output_path.with_suffix(f".{target_format}")

        try:
            # Soundfile typically handles float32 or int16. Ensure data type is appropriate.
            # Normalize and convert to float32 if not already
            audio_to_save = audio.astype(np.float32)
            max_val = np.abs(audio_to_save).max()
            if max_val > 1.0: # Clip if post-processing resulted in values > 1.0
                 audio_to_save /= max_val
                 self.logger.warning("Audio clipped during saving due to values > 1.0.")

            # Specify subtype for WAV if needed, otherwise let soundfile choose based on format
            subtype = 'PCM_16' if target_format == "wav" else None

            sf.write(
                output_path,
                audio_to_save,
                self.config.sample_rate,
                subtype=subtype
                # Add format argument if subtype isn't enough, e.g., format=target_format.upper()
            )
            self.logger.info(f"Audio saved to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to save audio to {output_path}: {e}")
            return None


# --- Local Hugging Face Backend ---

class LocalAudioBackend(AudioBackend):
    """Generate audio using local Hugging Face Transformers models."""

    SUPPORTED_MODELS = { # Keep track of known loading methods
        "speecht5": ["microsoft/speecht5_tts"],
        "bark": ["suno/bark", "suno/bark-small"],
        "mms": ["facebook/mms-tts-eng"],
        "vits": ["facebook/mms-tts-eng"], # MMS uses VITS architecture
        "tacotron2": ["speechbrain/tts-tacotron2-ljspeech"],
        "fastspeech2": ["facebook/fastspeech2-en-ljspeech"],
        "vibevoice": ["VibeVoice", "vibevoice"],
    }

    def __init__(self, config: AudioConfig, model_specifier: str):
        super().__init__(config, model_specifier)
        self.model = None
        self.processor = None
        self.vocoder = None # Specific to SpeechT5
        self.pipeline = None # For pipeline-based models
        self.device = self._setup_device()
        self.model_downloader = ModelDownloader() # Handles HF downloads
        # Lazy load datasets for speaker embeddings if needed
        self.speaker_embeddings = None
        self.embeddings_dataset = None

    @property
    def provider_identifier(self) -> str:
        return "local"

    @log_execution_time()
    def load_model(self, **kwargs) -> bool:
        """Load local TTS model from Hugging Face."""
        model_id = self.model_specifier
        self.logger.info(f"Loading local TTS model: {model_id} onto {self.device}")

        try:
            # Ensure model is downloaded
            model_path = self.model_downloader.download_model(model_id)
            if not model_path:
                self.logger.error(f"Failed to download model: {model_id}")
                return False

            # Detect model type and load
            model_id_lower = model_id.lower()
            loaded = False
            if any(term in model_id_lower for term in self.SUPPORTED_MODELS["bark"]):
                self._load_bark_model(model_id)
                loaded = True
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["speecht5"]):
                self._load_speecht5_model(model_id)
                loaded = True
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["mms"]):
                 # Assuming MMS uses VITS architecture
                self._load_vits_model(model_id)
                loaded = True
            elif any(term.lower() in model_id_lower for term in self.SUPPORTED_MODELS["vibevoice"]):
                self.logger.info("Detected VibeVoice model, using enhanced pipeline loading")
                self._load_generic_pipeline(model_id)
                loaded = True
            # Add checks for Tacotron2, FastSpeech2 if using direct loading instead of pipeline
            # elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["tacotron2"]):
            #     # ... specific loading ...
            # elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["fastspeech2"]):
            #     # ... specific loading ...

            if not loaded:
                # Fallback: Try generic pipeline loading
                self.logger.info(f"Attempting to load {model_id} using generic TTS pipeline.")
                self._load_generic_pipeline(model_id)

            self.logger.info(f"Model loaded successfully: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            # Reset components
            self.model = self.processor = self.vocoder = self.pipeline = None
            return False

    def _load_bark_model(self, model_id: str):
        from transformers import BarkModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if self.config.use_half_precision and self.device == "cuda" else torch.float32
        self.model = BarkModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.logger.info(f"Bark model loaded with dtype: {dtype}")

    def _load_speecht5_model(self, model_id: str):
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset # Import here

        self.processor = SpeechT5Processor.from_pretrained(model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_id).to(self.device)
        # Vocoder is essential for SpeechT5 quality
        try:
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        except Exception as e:
            self.logger.warning(f"Could not load SpeechT5 HiFiGan vocoder: {e}. Audio quality may be lower.")
            self.vocoder = None

        # Load speaker embeddings (only if not already loaded)
        if self.speaker_embeddings is None:
             try:
                 self.logger.info("Loading speaker embeddings dataset (cmu-arctic-xvectors)...")
                 self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                 # Default speaker - can be made configurable
                 default_speaker_idx = 7306 # Example index
                 self.speaker_embeddings = torch.tensor(
                     self.embeddings_dataset[default_speaker_idx]["xvector"]
                 ).unsqueeze(0).to(self.device)
                 self.logger.info(f"Using default speaker embedding index: {default_speaker_idx}")
             except Exception as e:
                 self.logger.error(f"Failed to load speaker embeddings: {e}. SpeechT5 may not work correctly.")
                 self.speaker_embeddings = None # Ensure generation fails if embeddings required but missing


    def _load_vits_model(self, model_id: str):
        # Covers MMS and potentially other VITS models
        from transformers import VitsModel, AutoTokenizer
        self.processor = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(self.device)
        # Ensure model has 'forward' method for generation or adapt call
        if not hasattr(self.model, 'forward'):
             # VITS models might have a specific generation method
             # For MMS, the pipeline handles it, but direct use might need adjustment
             self.logger.warning(f"Loaded VITS model {model_id} might require specific generation call.")


    def _load_generic_pipeline(self, model_id: str):
        """Load generic TTS model using pipeline."""
        # Determine device index for pipeline
        pipeline_device = 0 if self.device == "cuda" else -1

        self.logger.info(f"Initializing TTS pipeline for {model_id}...")
        
        try:
            self.pipeline = pipeline(
                "text-to-speech",
                model=model_id,
                device=pipeline_device,
                trust_remote_code=True
            )

            # Store the pipeline's internal model/processor if needed, or rely on pipeline call
            if hasattr(self.pipeline, 'model') and self.pipeline.model:
                self.model = self.pipeline.model
            if hasattr(self.pipeline, 'tokenizer') and self.pipeline.tokenizer:
                self.processor = self.pipeline.tokenizer # Use processor for consistency
            elif hasattr(self.pipeline, 'feature_extractor') and self.pipeline.feature_extractor:
                self.processor = self.pipeline.feature_extractor

            self.logger.info(f"Pipeline initialized successfully for {model_id}")

        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    @log_execution_time()
    def generate_audio(self,
                      text: str,
                      output_path: Optional[Path] = None,
                      chunk_text: bool = True,
                      **kwargs) -> Optional[AudioResult]:
        """Generate audio from text using the loaded local model."""
        if not self.model and not self.pipeline:
            self.logger.error("No local model or pipeline loaded.")
            return None

        # Determine default output path if none provided
        if output_path is None:
             timestamp = time.strftime("%Y%m%d_%H%M%S")
             output_path = Path(f"output_audio_{timestamp}.{self.config.output_format}")
        else:
             output_path = Path(output_path) # Ensure it's a Path object

        self.logger.info(f"Generating audio for {len(text)} characters to {output_path.name}")
        start_time = time.time()

        try:
            # Process in chunks if text is long and chunking is enabled
            audio_arrays = []
            chunks_processed = 0
            if chunk_text and len(text) > self.config.chunk_size:
                text_chunks = self._split_text(text)
                chunks_processed = len(text_chunks)
                self.logger.info(f"Text too long, splitting into {chunks_processed} chunks.")
                try:
                    with LoggingProgress(self.logger, "Generating audio chunks", chunks_processed) as progress:
                        for i, chunk in enumerate(text_chunks):
                            self.logger.debug(f"Processing chunk {i+1}/{chunks_processed} (len: {len(chunk)})")
                            audio_chunk = self._generate_single_chunk(chunk)
                            if audio_chunk is not None and audio_chunk.size > 0:
                                audio_arrays.append(audio_chunk)
                            else:
                                 self.logger.warning(f"Chunk {i+1} generated empty audio, skipping.")
                            progress.update(1)
                except ImportError:
                    # Fallback without a progress bar
                    for i, chunk in enumerate(text_chunks):
                        self.logger.debug(f"Processing chunk {i+1}/{chunks_processed}")
                        audio_chunk = self._generate_single_chunk(chunk)
                        if audio_chunk is not None and audio_chunk.size > 0:
                            audio_arrays.append(audio_chunk)
            else:
                audio_chunk = self._generate_single_chunk(text)
                if audio_chunk is not None and audio_chunk.size > 0:
                     audio_arrays = [audio_chunk]
                chunks_processed = 1

            if not audio_arrays:
                 self.logger.error("No valid audio generated from any chunks.")
                 return None

            # Combine audio chunks
            combined_audio = self._combine_audio(audio_arrays)
            if combined_audio is None or combined_audio.size == 0:
                self.logger.error("Failed to combine audio chunks or result is empty.")
                return None

            # Post-process audio
            processed_audio = self._post_process_audio(combined_audio)

            # Save audio
            saved_path = self._save_audio(processed_audio, output_path)
            if not saved_path:
                return None # Saving failed

            # Calculate statistics
            duration = len(processed_audio) / self.config.sample_rate
            processing_time = time.time() - start_time

            result = AudioResult(
                audio_path=saved_path,
                duration_seconds=duration,
                sample_rate=self.config.sample_rate, # Assuming post-processing didn't change sr
                num_samples=len(processed_audio),
                model_used=self.model_identifier,
                processing_time=processing_time,
                chunks_processed=chunks_processed
            )

            self.logger.info(f"Audio generated: {result.duration_formatted} duration, model: {result.model_used}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate audio: {e}", exc_info=True)
            return None

    def _generate_single_chunk(self, text: str) -> Optional[np.ndarray]:
        """Generate audio for a single text chunk using the appropriate method."""
        try:
            # Check model type and call corresponding generation function
            model_id_lower = self.model_specifier.lower()

            if self.pipeline:
                # Use pipeline if it was loaded (generic or specific like Tacotron2)
                return self._generate_pipeline(text)
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["bark"]):
                return self._generate_bark(text)
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["speecht5"]):
                return self._generate_speecht5(text)
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["mms"]):
                return self._generate_vits(text) # Assuming MMS uses VITS
#            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["vibevoice"]):
#                return self._generate_vibevoice(text)
            # Add other specific model calls here if needed
            else:
                self.logger.error(f"Generation logic not implemented for model type: {self.model_specifier}. Try pipeline.")
                # Attempt pipeline as a last resort if not already tried
                if not self.pipeline:
                     self._load_generic_pipeline(self.model_specifier)
                     if self.pipeline:
                          return self._generate_pipeline(text)
                return None
        except Exception as e:
             self.logger.error(f"Error generating audio chunk for '{text[:50]}...': {e}", exc_info=True)
             return None


    # --- Specific Model Generation Methods ---
    def _generate_speecht5(self, text: str) -> Optional[np.ndarray]:
        if not self.model or not self.processor or self.speaker_embeddings is None:
            self.logger.error("SpeechT5 components not fully loaded (model, processor, or embeddings missing).")
            return None
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # SpeechT5 requires vocoder for high quality
            if self.vocoder:
                speech = self.model.generate_speech(
                    inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
                )
            else:
                 # Generate spectrogram and handle vocoding manually or accept lower quality
                 spectrogram = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings)
                 # If no vocoder, the output might be spectrogram, not waveform.
                 # This needs handling - either error out or try a default Griffin-Lim vocoder (low quality)
                 self.logger.warning("No vocoder loaded for SpeechT5. Returning raw output (may be spectrogram).")
                 # For now, let's assume generate_speech returns waveform even without vocoder (might be low quality)
                 speech = spectrogram # Placeholder - check actual SpeechT5 output without vocoder

        self.config.sample_rate = 16000 # SpeechT5 standard
        return speech.cpu().numpy().squeeze()

    def _generate_vits(self, text: str) -> Optional[np.ndarray]:
        # Covers MMS and potentially other VITS models
        if not self.model or not self.processor: return None
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # VITS models typically output waveform directly
            output = self.model(**inputs)
            # Access waveform based on model output structure (might be output.waveform or similar)
            waveform = output.waveform if hasattr(output, 'waveform') else output[0] # Adjust based on actual output
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'sampling_rate'):
            self.config.sample_rate = self.model.config.sampling_rate # Get from model config
        else:
            self.logger.warning("Could not determine sample rate from VITS model, using config default")
        return waveform.cpu().numpy().squeeze()

    def _generate_pipeline(self, text: str) -> Optional[np.ndarray]:
        """Generate using the loaded Transformers pipeline - ENHANCED."""
        if not self.pipeline: return None
        try:
            # Pipeline expects string input, returns dict or audio data
            self.logger.debug(f"Calling pipeline for text: {text[:50]}...")
            output = self.pipeline(text)

            # FIXED: Enhanced output parsing to handle more formats
            audio_array = None
            detected_sr = None
            
            # Try multiple output format possibilities
            if isinstance(output, dict):
                # Format 1: {"audio": array, "sampling_rate": int}
                if "audio" in output:
                    audio_array = output["audio"]
                    detected_sr = output.get("sampling_rate") or output.get("sample_rate")
                # Format 2: {"waveform": array, "sample_rate": int}
                elif "waveform" in output:
                    audio_array = output["waveform"]
                    detected_sr = output.get("sample_rate") or output.get("sampling_rate")
                # Format 3: Keys might be different for VibeVoice
                elif "generated_audio" in output:
                    audio_array = output["generated_audio"]
                    detected_sr = output.get("sr") or output.get("sampling_rate")
                else:
                    # Try to find array-like values
                    for key, value in output.items():
                        if isinstance(value, (np.ndarray, list)) and not key.startswith('_'):
                            audio_array = value
                            self.logger.info(f"Found audio data under key: {key}")
                            break
                    
            elif isinstance(output, (np.ndarray, list)):
                 # Direct array output
                 audio_array = np.array(output)
                 
            elif isinstance(output, tuple):
                # Some models return (audio, sample_rate)
                if len(output) >= 2:
                    audio_array = output[0]
                    detected_sr = output[1] if isinstance(output[1], (int, float)) else None
                elif len(output) == 1:
                    audio_array = output[0]
            else:
                 self.logger.error(f"Unexpected output format from pipeline: {type(output)}")
                 self.logger.debug(f"Output keys/type: {output.keys() if isinstance(output, dict) else type(output)}")
                 return None

            # Ensure we have audio data
            if audio_array is None:
                self.logger.error("Could not extract audio array from pipeline output")
                return None
                
            # Convert to numpy array if needed
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            
            # Update sample rate if detected
            if detected_sr is not None:
                self.config.sample_rate = int(detected_sr)
                self.logger.info(f"Detected sample rate from pipeline: {self.config.sample_rate}Hz")
            else:
                # FIXED: Try to get from model config as fallback
                if hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model, 'config'):
                    model_config = self.pipeline.model.config
                    for sr_attr in ['sampling_rate', 'sample_rate', 'sr']:
                        if hasattr(model_config, sr_attr):
                            self.config.sample_rate = int(getattr(model_config, sr_attr))
                            self.logger.info(f"Got sample rate from model config: {self.config.sample_rate}Hz")
                            break
                else:
                    self.logger.warning(f"Could not determine sample rate from pipeline output. Using config default: {self.config.sample_rate}Hz")

            # Squeeze unnecessary dimensions
            audio_array = audio_array.squeeze()
            
            # Validate output
            if audio_array.size == 0:
                self.logger.error("Pipeline returned empty audio array")
                return None
                
            self.logger.debug(f"Successfully generated audio: shape={audio_array.shape}, dtype={audio_array.dtype}")
            return audio_array

        except Exception as e:
            self.logger.error(f"Error during pipeline generation: {e}", exc_info=True)
            return None


    def unload_model(self):
        """Unload local model and free memory."""
        self.logger.info(f"Unloading local model: {self.model_specifier}")
        if self.model: del self.model
        if self.processor: del self.processor
        if self.vocoder: del self.vocoder
        if self.pipeline: del self.pipeline
        self.model = self.processor = self.vocoder = self.pipeline = None
        self.speaker_embeddings = None # Clear embeddings too
        self.embeddings_dataset = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Local model unloaded.")

    def __del__(self):
        self.unload_model()

# --- Cloud Backend Example: OpenAI TTS ---

# Try importing OpenAI library
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

class OpenAITTSBackend(AudioBackend):
    """Generate audio using OpenAI's Text-to-Speech API."""

    def __init__(self, config: AudioConfig, model_specifier: str, api_key: str):
        super().__init__(config, model_specifier)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run 'pip install openai'")
        self.api_key = api_key
        # Model specifier should be 'tts-1' or 'tts-1-hd'
        if model_specifier not in ["tts-1", "tts-1-hd"]:
             self.logger.warning(f"Unsupported OpenAI model '{model_specifier}', defaulting to 'tts-1'.")
             self.model_specifier = "tts-1"
        self.client = openai.OpenAI(api_key=self.api_key)
        self.voice = self.config.cloud_voice # Use voice from config

    @property
    def provider_identifier(self) -> str:
        return "openai"

    def load_model(self, **kwargs) -> bool:
        # No model loading needed for API
        self.logger.info("OpenAI TTS backend initialized (no model loading required).")
        return True

    def unload_model(self):
        # No unloading needed for API
        pass

    @log_execution_time()
    def generate_audio(self,
                      text: str,
                      output_path: Optional[Path] = None,
                      chunk_text: bool = True, # OpenAI API handles long text, but chunking might still be wise for very long inputs
                      **kwargs) -> Optional[AudioResult]:
        """Generate audio using OpenAI API."""
        if output_path is None:
             timestamp = time.strftime("%Y%m%d_%H%M%S")
             output_path = Path(f"output_audio_{timestamp}.{self.config.output_format}")
        else:
             output_path = Path(output_path)

        # OpenAI TTS preferred format is often mp3, adjust suffix
        # Let _save_audio handle the final format conversion if possible,
        # but ask OpenAI for a format it supports directly.
        api_format = "mp3" # Or check config if API supports wav/flac directly
        temp_output_path = output_path.with_suffix(f".{api_format}")


        self.logger.info(f"Generating audio via OpenAI TTS (model: {self.model_specifier}, voice: {self.voice})")
        start_time = time.time()

        try:
            # OpenAI API call - handles input directly, saves to file path
            response = self.client.audio.speech.create(
                model=self.model_specifier,
                voice=self.voice,
                input=text,
                response_format=api_format,
                speed=self.config.speed # API supports speed directly
                # Pitch/Normalization are not direct API params, need post-processing
            )
            # Stream response directly to the temporary file
            response.stream_to_file(str(temp_output_path))
            processing_time = time.time() - start_time
            self.logger.info(f"OpenAI TTS API call successful (took {processing_time:.2f}s)")

            # Post-process (pitch, normalize) and convert format if necessary
            final_audio_path = self._handle_postprocessing_and_format(temp_output_path, output_path)
            if not final_audio_path:
                 return None # Post-processing or saving failed


            # Get audio info from the final saved file
            try:
                audio_info = sf.info(str(final_audio_path))
                duration = audio_info.duration
                sample_rate = audio_info.samplerate
                num_samples = audio_info.frames
                # Update config sample rate if needed (OpenAI usually uses 24kHz)
                if self.config.sample_rate != sample_rate:
                     self.logger.info(f"Actual sample rate is {sample_rate}Hz (config was {self.config.sample_rate}Hz).")
                     # Decide whether to resample here or just report the actual rate
                     # For now, report actual rate. Resampling could be added.
                     actual_sr = sample_rate
                else:
                     actual_sr = self.config.sample_rate

            except Exception as info_err:
                 self.logger.warning(f"Could not read info from saved audio file {final_audio_path}: {info_err}")
                 duration, actual_sr, num_samples = 0.0, self.config.sample_rate, 0 # Fallback

            result = AudioResult(
                audio_path=final_audio_path,
                duration_seconds=duration,
                sample_rate=actual_sr,
                num_samples=num_samples,
                model_used=self.model_identifier,
                processing_time=processing_time,
                chunks_processed=1 # API handles input length
            )

            self.logger.info(f"Audio generated: {result.duration_formatted} duration, model: {result.model_used}")
            return result

        except Exception as e:
            self.logger.error(f"OpenAI TTS API call failed: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_output_path.exists():
                 try: temp_output_path.unlink()
                 except OSError: pass
            return None

    def _handle_postprocessing_and_format(self, input_audio_path: Path, target_output_path: Path) -> Optional[Path]:
         """Apply post-processing and ensure final format."""
         target_format = self.config.output_format.lower()
         input_format = input_audio_path.suffix[1:].lower()

         needs_postprocessing = (self.config.pitch_shift != 0 or self.config.volume_normalize)
         needs_format_conversion = (input_format != target_format)

         if not needs_postprocessing and not needs_format_conversion:
              # Just rename if necessary
              if input_audio_path != target_output_path:
                   try:
                        input_audio_path.rename(target_output_path)
                        return target_output_path
                   except OSError as e:
                        self.logger.error(f"Failed to rename audio file: {e}")
                        return None
              return input_audio_path

         # Load audio for processing/conversion
         try:
              audio, sr = librosa.load(input_audio_path, sr=None) # Load with original SR
              # Check if API SR differs from config SR - decide if resampling is needed
              if sr != self.config.sample_rate:
                   self.logger.info(f"Resampling audio from {sr}Hz to {self.config.sample_rate}Hz.")
                   audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
                   current_sr = self.config.sample_rate
              else:
                   current_sr = sr

              processed_audio = audio

              # Apply post-processing (only those not handled by API)
              if self.config.pitch_shift != 0:
                   processed_audio = librosa.effects.pitch_shift(
                       y=processed_audio.astype(np.float32), sr=current_sr, n_steps=self.config.pitch_shift
                   )
              if self.config.volume_normalize:
                   processed_audio = processed_audio.astype(np.float32)
                   max_val = np.abs(processed_audio).max()
                   if max_val > 1e-6: processed_audio = processed_audio / max_val * 0.95

              # Ensure correct suffix for saving
              final_output_path = target_output_path.with_suffix(f".{target_format}")

              # Save in the target format
              saved_path = self._save_audio(processed_audio, final_output_path) # Use parent class save method

              # Clean up temporary input file if different from final path
              if input_audio_path != final_output_path and input_audio_path.exists():
                   try: input_audio_path.unlink()
                   except OSError: pass

              return saved_path

         except Exception as e:
              self.logger.error(f"Error during audio post-processing/conversion: {e}", exc_info=True)
              # Clean up temp file
              if input_audio_path.exists() and input_audio_path != target_output_path:
                   try: input_audio_path.unlink()
                   except OSError: pass
              return None

# --- Factory Function ---

def get_audio_backend(provider: str,
                      model_specifier: str,
                      api_keys: Dict[str, str],
                      config: AudioConfig) -> Optional[AudioBackend]:
    """Factory function to create the appropriate audio backend."""
    provider_lower = provider.lower()
    logger.info(f"Attempting to initialize audio backend for provider: {provider_lower}, model: {model_specifier}")

    try:
        if provider_lower == "local":
            return LocalAudioBackend(config=config, model_specifier=model_specifier)
        elif provider_lower == "openai":
            api_key = api_keys.get("openai")
            if not api_key:
                logger.error("OpenAI API key not found in configuration.")
                return None
            return OpenAITTSBackend(config=config, model_specifier=model_specifier, api_key=api_key)
        # --- Add other cloud providers here ---
        # elif provider_lower == "google":
        #     api_key = api_keys.get("google")
        #     if not api_key: logger.error(...); return None
        #     return GoogleTTSBackend(...) # Requires implementation
        # elif provider_lower == "anthropic": # Check if they offer TTS
        #     logger.warning("Anthropic does not currently offer a public TTS API.")
        #     return None
        # ... etc. ...
        else:
            logger.error(f"Unsupported audio provider: {provider}")
            return None
    except ImportError as e:
         logger.error(f"Missing library for audio provider '{provider_lower}': {e}. Please install it.")
         return None
    except Exception as e:
         logger.error(f"Failed to initialize audio backend for {provider}: {e}", exc_info=True)
         return None


# --- Audio Post Processing Class --- (Keep as is)
class AudioPostProcessor:
    """Post-process audio files (add music, silence, effects)."""
    @staticmethod
    def add_background_music(speech_path: Path,
                            music_path: Path,
                            output_path: Path,
                            music_volume: float = 0.2) -> Optional[Path]:
        """Add background music to speech."""
        try:
            speech, sr1 = librosa.load(speech_path, sr=None)
            music, sr2 = librosa.load(music_path, sr=None)

            # Resample music if sample rates differ
            if sr1 != sr2:
                logger.info(f"Resampling background music from {sr2}Hz to {sr1}Hz.")
                music = librosa.resample(music, orig_sr=sr2, target_sr=sr1)

            # Ensure music is at least as long as speech (loop if necessary)
            if len(music) < len(speech):
                repeats = int(np.ceil(len(speech) / len(music)))
                music = np.tile(music, repeats)

            # Trim music to match speech length
            music = music[:len(speech)]

            # Mix audio
            mixed_audio = speech + (music * music_volume)

            # Normalize mixed audio to prevent clipping
            max_val = np.abs(mixed_audio).max()
            if max_val > 1e-6:
                mixed_audio = mixed_audio / max_val * 0.95
            else:
                 logger.warning("Mixed audio is silent, skipping normalization.")
                 mixed_audio = np.zeros_like(speech) # Output silence if input was silent

            # Save the mixed audio
            sf.write(output_path, mixed_audio, sr1)
            logger.info(f"Background music added. Output saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to add background music: {e}", exc_info=True)
            return None


    @staticmethod
    def add_silence(audio_path: Path,
                   output_path: Optional[Path] = None, # Allow separate output
                   start_silence: float = 0.5,
                   end_silence: float = 1.0) -> Optional[Path]:
        """Add silence to the beginning and end of an audio file."""
        try:
            audio, sr = librosa.load(audio_path, sr=None)

            start_samples = int(start_silence * sr)
            end_samples = int(end_silence * sr)

            # Create silence arrays
            start_padding = np.zeros(start_samples, dtype=audio.dtype)
            end_padding = np.zeros(end_samples, dtype=audio.dtype)

            # Concatenate
            padded_audio = np.concatenate([start_padding, audio, end_padding])

            # Determine output path
            save_path = output_path or audio_path # Overwrite if no output path given

            # Save the modified audio
            sf.write(save_path, padded_audio, sr)
            logger.info(f"Added silence. Output saved to: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to add silence: {e}", exc_info=True)
            return None

    @staticmethod
    def apply_effects(audio_path: Path,
                     output_path: Optional[Path] = None, # Allow separate output
                     reverb: float = 0.0, # Reverb amount (e.g., 0.1 for slight reverb)
                     echo_delay: float = 0.0, # Echo delay in seconds
                     echo_decay: float = 0.5) -> Optional[Path]: # Echo feedback decay (0-1)
        """Apply simple reverb and/or echo effects."""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            processed_audio = audio.copy().astype(np.float32) # Work with float

            # Simple Reverb (Convolution with exponential decay IR) - Basic approximation
            if reverb > 0.0:
                logger.info(f"Applying reverb (amount: {reverb:.2f})")
                try:
                     # This is a very basic reverb, consider libraries like pedalboard for better quality
                     from scipy.signal import convolve
                     decay_time = reverb * 2 # Heuristic: decay time related to reverb amount
                     ir_length = int(decay_time * sr)
                     if ir_length > 0:
                          t = np.linspace(0, decay_time, ir_length)
                          # Exponential decay impulse response
                          impulse_response = np.exp(-5 * t / decay_time) * np.random.randn(ir_length)
                          impulse_response /= np.sum(np.abs(impulse_response)) # Normalize
                          processed_audio = convolve(processed_audio, impulse_response, mode='same')
                except ImportError:
                     logger.warning("Scipy not installed, cannot apply reverb effect.")
                except Exception as reverb_err:
                     logger.error(f"Error applying reverb: {reverb_err}")


            # Simple Echo (Feedback delay)
            if echo_delay > 0.0 and 0.0 < echo_decay < 1.0:
                logger.info(f"Applying echo (delay: {echo_delay:.2f}s, decay: {echo_decay:.2f})")
                try:
                     delay_samples = int(echo_delay * sr)
                     if delay_samples > 0 and delay_samples < len(processed_audio):
                          echo_component = np.zeros_like(processed_audio)
                          # Add delayed signal with decay
                          echo_component[delay_samples:] = processed_audio[:-delay_samples] * echo_decay
                          # Simple single echo, could be made recursive for feedback echo
                          processed_audio = processed_audio + echo_component
                     else:
                          logger.warning("Echo delay is too long or zero, skipping echo effect.")
                except Exception as echo_err:
                     logger.error(f"Error applying echo: {echo_err}")

            # Normalize final audio
            max_val = np.abs(processed_audio).max()
            if max_val > 1e-6:
                processed_audio = processed_audio / max_val * 0.95
            else:
                 logger.warning("Audio after effects is silent.")
                 processed_audio = np.zeros_like(audio)

            # Determine output path
            save_path = output_path or audio_path

            # Save the processed audio
            sf.write(save_path, processed_audio, sr)
            logger.info(f"Applied effects. Output saved to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to apply audio effects: {e}", exc_info=True)
            return None
