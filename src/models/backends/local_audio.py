# llamanote/models/backends/local_audio.py
"""
Local Audio (TTS) Backend
Generates audio using local Hugging Face Transformers models.
"""

import gc
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import AudioBackend
from ...core.types import AudioConfig, AudioResult
from ...core.errors import ModelLoadError, GenerationError
from ...utils.logger import get_logger_conf, log_execution_time, LoggingProgress
from ...utils.helpers import get_device_manager, cleanup_resources
from ...models.hub import ModelHub
from ...processing.audio_processor import AudioPostProcessor # Import post-processor
from ...config.settings import CACHE_DIR

# --- Lazy Imports for transformers components ---
_AutoProcessor = None
_AutoModel = None
_pipeline = None
_SpeechT5Processor = None
_SpeechT5ForTextToSpeech = None
_SpeechT5HifiGan = None
_BarkModel = None
_VitsModel = None
_AutoTokenizer = None

def _import_transformers():
    global _AutoProcessor, _AutoModel, _pipeline
    global _SpeechT5Processor, _SpeechT5ForTextToSpeech, _SpeechT5HifiGan
    global _BarkModel, _VitsModel, _AutoTokenizer
    
    if _AutoProcessor is None:
        try:
            from transformers import (
                AutoProcessor, AutoModel, pipeline,
                SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                BarkModel, VitsModel, AutoTokenizer
            )
            _AutoProcessor = AutoProcessor
            _AutoModel = AutoModel
            _pipeline = pipeline
            _SpeechT5Processor = SpeechT5Processor
            _SpeechT5ForTextToSpeech = SpeechT5ForTextToSpeech
            _SpeechT5HifiGan = SpeechT5HifiGan
            _BarkModel = BarkModel
            _VitsModel = VitsModel
            _AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError("Hugging Face 'transformers' library not found. Please install it: pip install transformers")

def _import_datasets():
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        logger.error("Hugging Face 'datasets' library not found. Required for SpeechT5 speaker embeddings.")
        raise ImportError("Hugging Face 'datasets' library not found. Install with: pip install datasets")


logger = get_logger_conf(__name__)

class LocalAudioBackend(AudioBackend):
    """Generate audio using local Hugging Face Transformers models."""

    SUPPORTED_MODELS = {
        "speecht5": ["microsoft/speecht5_tts"],
        "bark": ["suno/bark", "suno/bark-small"],
        "mms": ["facebook/mms-tts-eng"],
        "vits": ["facebook/mms-tts-eng"], # MMS uses VITS
        "vibevoice": ["vibevoice/vibevoice"],
    }

    def __init__(self, config: AudioConfig, model_specifier: str):
        super().__init__(config, model_specifier)
        _import_transformers() # Ensure transformers is available
        
        self.model = None
        self.processor = None
        self.vocoder = None # Specific to SpeechT5
        self.pipeline = None # For pipeline-based models
        self.device = get_device_manager().get_device()
        
        self.model_hub = ModelHub(cache_dir=CACHE_DIR / "audio_models")
        self.speaker_embeddings = None
        self.embeddings_dataset = None
        self.post_processor = AudioPostProcessor() # Use the separated class

    @property
    def provider_identifier(self) -> str:
        return "local_audio"

    @log_execution_time(logger_name=__name__)
    def load(self, **kwargs) -> bool:
        """Load local TTS model from Hugging Face."""
        if self.model_handle is not None:
            self.logger.info("Local audio model already loaded.")
            return True
            
        model_id = self.model_specifier
        self.logger.info(f"Loading local TTS model: {model_id} onto {self.device}")
        ConsoleOutput.info(f"Loading local audio model: {model_id}...")

        try:
            # Check if model is already cached/downloaded
            if not self.model_hub.is_model_cached(model_id): # This checks registry, might need to check hub cache path
                self.logger.info(f"Model not cached, downloading: {model_id}")
                # Note: model_hub.download_model uses snapshot_download
                model_path = self.model_hub.download_model(model_id, 
                                                            cache_dir=self.model_hub.model_cache_dir)
                if not model_path:
                    raise ModelLoadError(f"Failed to download model: {model_id}")
                self.logger.info(f"Model downloaded to: {model_path}")
            else:
                self.logger.info(f"Using cached model: {model_id}")

            # Detect model type and load
            model_id_lower = model_id.lower()
            loaded = False
            
            # --- Model Loading Strategy ---
            if any(term in model_id_lower for term in self.SUPPORTED_MODELS["bark"]):
                self._load_bark_model(model_id)
                loaded = True
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["speecht5"]):
                self._load_speecht5_model(model_id)
                loaded = True
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["mms"]):
                self._load_vits_model(model_id)
                loaded = True
            elif any(term.lower() in model_id_lower for term in self.SUPPORTED_MODELS["vibevoice"]):
                self._load_generic_pipeline(model_id)
                loaded = True

            if not loaded:
                self.logger.info(f"Attempting to load {model_id} using generic TTS pipeline.")
                self._load_generic_pipeline(model_id)

            # Assign to the base class handle
            self.model_handle = self.model or self.pipeline
            
            if self.model_handle is None:
                 raise ModelLoadError(f"Model loading failed, no model or pipeline object was created for {model_id}.")

            self.logger.info(f"Model loaded successfully: {model_id}")
            ConsoleOutput.success("Audio model loaded.")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            self.unload() # Ensure cleanup on failure
            raise ModelLoadError(f"Failed to load model {model_id}: {e}", model_id) from e

    def unload(self):
        """Unload local model and free memory."""
        self.logger.info(f"Unloading local audio model: {self.model_specifier}")
        # Use resource cleanup helper
        cleanup_resources(
            [self.model, self.processor, self.vocoder, self.pipeline, self.embeddings_dataset],
            clear_cuda=(self.device == 'cuda')
        )
        self.model = None
        self.processor = None
        self.vocoder = None
        self.pipeline = None
        self.model_handle = None # Clear base class handle
        self.speaker_embeddings = None
        self.embeddings_dataset = None
        self.logger.info("Local audio model unloaded.")

    def _load_bark_model(self, model_id: str):
        self.processor = _AutoProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR)
        dtype = torch.float16 if self.config.use_half_precision and self.device == "cuda" else torch.float32
        self.model = _BarkModel.from_pretrained(model_id, torch_dtype=dtype, cache_dir=CACHE_DIR).to(self.device)
        self.config.sample_rate = self.model.generation_config.sample_rate # Get SR from model
        self.logger.info(f"Bark model loaded with dtype: {dtype}, Sample Rate: {self.config.sample_rate}Hz")

    def _load_speecht5_model(self, model_id: str):
        load_dataset = _import_datasets() # Ensure datasets is available
        
        self.processor = _SpeechT5Processor.from_pretrained(model_id, cache_dir=CACHE_DIR)
        self.model = _SpeechT5ForTextToSpeech.from_pretrained(model_id, cache_dir=CACHE_DIR).to(self.device)
        
        # Vocoder is essential for SpeechT5
        try:
            vocoder_id = "microsoft/speecht5_hifigan"
            self.vocoder = _SpeechT5HifiGan.from_pretrained(vocoder_id, cache_dir=CACHE_DIR).to(self.device)
        except Exception as e:
            self.logger.warning(f"Could not load SpeechT5 HiFiGan vocoder: {e}. Audio quality may be low.")
            self.vocoder = None

        # Load speaker embeddings
        if self.config.speaker_embedding:
            self.logger.warning("Custom speaker embeddings not yet supported, using default.")
            # TODO: Add logic to load custom embedding path/ID
            
        if self.speaker_embeddings is None:
             try:
                 self.logger.info("Loading default speaker embeddings (cmu-arctic-xvectors)...")
                 self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", cache_dir=CACHE_DIR / "datasets")
                 # Use a common, generic-sounding speaker
                 default_speaker_idx = 7306 # Example index
                 self.speaker_embeddings = torch.tensor(
                     self.embeddings_dataset[default_speaker_idx]["xvector"]
                 ).unsqueeze(0).to(self.device)
                 self.logger.info(f"Using default speaker embedding index: {default_speaker_idx}")
             except Exception as e:
                 self.logger.error(f"Failed to load speaker embeddings: {e}. SpeechT5 may not work.", exc_info=True)
                 raise ModelLoadError("Failed to load SpeechT5 speaker embeddings", model_id) from e

        self.config.sample_rate = 16000 # SpeechT5 standard

    def _load_vits_model(self, model_id: str):
        # Covers MMS and potentially other VITS models
        self.processor = _AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        self.model = _VitsModel.from_pretrained(model_id, cache_dir=CACHE_DIR).to(self.device)
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'sampling_rate'):
            self.config.sample_rate = self.model.config.sampling_rate
        else:
             self.config.sample_rate = 22050 # Common VITS default
        self.logger.info(f"VITS/MMS model loaded. Sample Rate: {self.config.sample_rate}Hz")


    def _load_generic_pipeline(self, model_id: str):
        """Load generic TTS model using pipeline with enhanced diagnostics."""
        pipeline_device_id = 0 if self.device == "cuda" else -1
        self.logger.info(f"Initializing TTS pipeline for {model_id} on device: {self.device} (ID: {pipeline_device_id})")
        
        try:
            self.pipeline = _pipeline(
                "text-to-speech",
                model=model_id,
                device=pipeline_device_id,
                trust_remote_code=True, # Required for many TTS models
                cache_dir=CACHE_DIR
            )
            
            # Try to get model info
            if hasattr(self.pipeline, 'model'):
                self.model = self.pipeline.model # Store model if accessible
                if hasattr(self.model, 'config'):
                    config = self.model.config
                    # Try to find sample rate in config
                    sr_attrs = ['sampling_rate', 'sample_rate', 'sr']
                    for attr in sr_attrs:
                        if hasattr(config, attr):
                            self.config.sample_rate = getattr(config, attr)
                            logger.info(f"Set sample rate from model config: {self.config.sample_rate}Hz")
                            break
            
            if hasattr(self.pipeline, 'tokenizer'):
                self.processor = self.pipeline.tokenizer
            elif hasattr(self.pipeline, 'feature_extractor'):
                self.processor = self.pipeline.feature_extractor
            
            self.logger.info(f"Pipeline initialized: {type(self.pipeline).__name__}")

        except Exception as e:
            self.logger.error(f"Failed to initialize generic TTS pipeline: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to initialize pipeline: {e}", model_id) from e

    @log_execution_time(logger_name=__name__)
    def generate_audio(self,
                      text: str,
                      output_path: Optional[Path] = None,
                      chunk_text: bool = True,
                      **kwargs) -> Optional[AudioResult]:
        """Generate audio from text using the loaded local model."""
        if self.model_handle is None:
            self.logger.error("No local model or pipeline loaded.")
            return None

        output_path = self._resolve_output_path(output_path)
        self.logger.info(f"Generating local audio for {len(text)} characters to {output_path.name}")
        start_time = time.time()

        try:
            # Split text into chunks if needed
            audio_arrays = []
            if chunk_text and len(text) > self.config.chunk_size:
                text_chunks = self._split_text(text)
                self.logger.info(f"Splitting text into {len(text_chunks)} chunks.")
                with LoggingProgress(self.logger, "Generating audio chunks", len(text_chunks)) as progress:
                    for i, chunk in enumerate(text_chunks):
                        audio_chunk = self._generate_single_chunk(chunk)
                        if audio_chunk is not None and audio_chunk.size > 0:
                            audio_arrays.append(audio_chunk)
                        progress.update(1)
            else:
                audio_chunk = self._generate_single_chunk(text)
                if audio_chunk is not None and audio_chunk.size > 0:
                    audio_arrays = [audio_chunk]

            if not audio_arrays:
                 raise GenerationError("No valid audio generated from any chunks.", self.model_specifier)

            # Combine chunks
            combined_audio = self._combine_audio(audio_arrays, self.config.sample_rate)
            if combined_audio is None or combined_audio.size == 0:
                raise GenerationError("Failed to combine audio chunks or result is empty.", self.model_specifier)

            # Post-process
            processed_audio = self.post_processor._post_process_audio(combined_audio, self.config.sample_rate, self.config.speed, self.config.pitch_shift, self.config.volume_normalize)

            # Save
            saved_path = self.post_processor._save_audio(output_path, processed_audio, self.config.sample_rate, self.config.output_format)
            if not saved_path:
                return None # Saving failed

            # Result
            duration = len(processed_audio) / self.config.sample_rate
            processing_time = time.time() - start_time
            result = AudioResult(
                audio_path=saved_path,
                duration_seconds=duration,
                sample_rate=self.config.sample_rate,
                num_samples=len(processed_audio),
                model_used=self.model_identifier,
                processing_time=processing_time,
                chunks_processed=len(audio_arrays)
            )
            self.logger.info(f"Local audio generated: {result.duration_formatted} duration")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate local audio: {e}", exc_info=True)
            # Ensure model handle is cleared if it's a critical error?
            return None # Propagate failure

    def _generate_single_chunk(self, text: str) -> Optional[np.ndarray]:
        """Generate audio for a single text chunk using the appropriate method."""
        try:
            model_id_lower = self.model_specifier.lower()

            if self.pipeline:
                return self._generate_with_pipeline(text)
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["bark"]):
                return self._generate_bark(text)
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["speecht5"]):
                return self._generate_speecht5(text)
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["mms"]):
                return self._generate_vits(text)
            else:
                self.logger.error(f"No specific generation logic found for {self.model_specifier}. Attempting pipeline.")
                self._load_generic_pipeline(self.model_specifier)
                if self.pipeline:
                     self.model_handle = self.pipeline # Update handle
                     return self._generate_with_pipeline(text)
                return None
        except Exception as e:
             self.logger.error(f"Error generating audio chunk for '{text[:50]}...': {e}", exc_info=True)
             return None

    # --- Specific Model Generation Methods ---
    
    def _generate_speecht5(self, text: str) -> Optional[np.ndarray]:
        if not self.model or not self.processor or self.speaker_embeddings is None:
            self.logger.error("SpeechT5 components not fully loaded.")
            return None
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.vocoder:
                speech = self.model.generate_speech(
                    inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
                )
            else:
                 spectrogram = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings)
                 self.logger.warning("No vocoder for SpeechT5. Output may be low quality or unusable.")
                 # This path likely fails if a vocoder is required to get waveform
                 speech = spectrogram # This is likely wrong
        
        self.config.sample_rate = 16000 # SpeechT5 fixed SR
        return speech.cpu().numpy().squeeze()

    def _generate_bark(self, text: str) -> Optional[np.ndarray]:
        if not self.model or not self.processor: return None
        # Bark generation often needs presets (e.g., "v2/en_speaker_6")
        voice_preset = "v2/en_speaker_6" # Make this configurable?
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
        
        # Bark output is usually 24kHz
        self.config.sample_rate = self.model.generation_config.sample_rate
        return audio_array.cpu().numpy().squeeze()

    def _generate_vits(self, text: str) -> Optional[np.ndarray]:
        if not self.model or not self.processor: return None
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            waveform = output.waveform if hasattr(output, 'waveform') else output[0]
        
        # SR should be in config from loading
        return waveform.cpu().numpy().squeeze()

    def _generate_with_pipeline(self, text: str) -> Optional[np.ndarray]:
        """Generate using the loaded Transformers pipeline."""
        if not self.pipeline: return None
        
        try:
            output = self.pipeline(text)
            
            # Handle various pipeline output formats
            audio_array = None
            detected_sr = None
            
            if isinstance(output, dict):
                audio_keys = ['audio', 'waveform']
                sr_keys = ['sampling_rate', 'sample_rate']
                
                for key in audio_keys:
                    if key in output:
                        audio_array = output[key]
                        break
                
                for key in sr_keys:
                    if key in output:
                        detected_sr = int(output[key])
                        break
                        
            elif isinstance(output, (np.ndarray, list)):
                audio_array = np.array(output) if isinstance(output, list) else output
                
            elif isinstance(output, tuple) and len(output) >= 2:
                 audio_array = output[0]
                 if isinstance(output[1], (int, float)):
                     detected_sr = int(output[1])

            if audio_array is None:
                self.logger.error(f"Unexpected pipeline output format: {type(output)}")
                return None
                
            # Convert to numpy array if it's a list
            if not isinstance(audio_array, np.ndarray):
                 audio_array = np.array(audio_array)

            # Squeeze to 1D if necessary
            audio_array = audio_array.squeeze()
            
            if audio_array.ndim > 1:
                 self.logger.warning(f"Audio array has unexpected shape {audio_array.shape}, taking mean of channels.")
                 audio_array = np.mean(audio_array, axis=0) # Simple stereo->mono
                 
            # Update config sample rate if detected
            if detected_sr is not None:
                self.config.sample_rate = detected_sr
            # Else, assume the SR set during load (or default) is correct
                
            return audio_array.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error during pipeline generation: {e}", exc_info=True)
            return None
