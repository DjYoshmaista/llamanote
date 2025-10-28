# llamanote/models/backends/local_audio.py
"""
Local (Hugging Face) Audio Generation Backend
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import AudioBackend
from ...core.types import AudioConfig, AudioResult
from ...utils.logger import get_logger_conf, LoggingProgress
from ...utils.decorators import log_execution_time
from ...models.hub import ModelHub
from ...config.settings import CACHE_DIR

# --- Lazy Imports for transformers components ---
# These will be imported only when this backend is actually used.
AutoProcessor = None
AutoModelForTextToSpeech = None
SpeechT5Processor = None
SpeechT5ForTextToSpeech = None
SpeechT5HifiGan = None
BarkModel = None
VitsModel = None
AutoTokenizer = None
pipeline = None
load_dataset = None
# --- End Lazy Imports ---

logger = get_logger_conf(__name__)

class LocalAudioBackend(AudioBackend):
    """
    Generate audio using local Hugging Face Transformers models.
    Supports various TTS models like SpeechT5, Bark, and VITS/MMS.
    """

    SUPPORTED_MODELS = {
        "speecht5": ["microsoft/speecht5_tts"],
        "bark": ["suno/bark", "suno/bark-small"],
        "mms": ["facebook/mms-tts-eng"],
        "vits": ["facebook/mms-tts-eng"], # MMS uses VITS architecture
    }

    def __init__(self, config: AudioConfig, model_specifier: str):
        super().__init__("local_audio", model_specifier, config)
        self._import_dependencies() # Ensure libraries are available

        self.processor = None
        self.vocoder = None # Specific to SpeechT5
        self.device = self._setup_device()
        
        # Use a specific cache directory for audio models
        self.model_downloader = ModelHub(cache_dir=CACHE_DIR / "audio_models") 
        
        # Speaker embeddings (for SpeechT5)
        self.speaker_embeddings = None
        self.embeddings_dataset = None

    def _import_dependencies(self):
        """Lazy-load required heavy libraries."""
        global AutoProcessor, AutoModelForTextToSpeech, SpeechT5Processor
        global SpeechT5ForTextToSpeech, SpeechT5HifiGan, BarkModel, VitsModel
        global AutoTokenizer, pipeline, load_dataset
        
        if AutoProcessor is None:
            from transformers import (
                AutoProcessor, AutoModelForTextToSpeech, SpeechT5Processor, 
                SpeechT5ForTextToSpeech, SpeechT5HifiGan, BarkModel, 
                VitsModel, AutoTokenizer, pipeline
            )
            from datasets import load_dataset


    def _setup_device(self) -> str:
        """Setup compute device (e.g., 'cuda' or 'cpu')."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                logger.info("CUDA is available. Setting device to 'cuda'.")
                return "cuda"
            else:
                logger.info("CUDA not available. Setting device to 'cpu'.")
                return "cpu"
        logger.info(f"Using specified device: {self.config.device}")
        return self.config.device

    @log_execution_time(logger_name=__name__)
    def load(self, **kwargs) -> bool:
        """Load local TTS model from Hugging Face."""
        if self.model_handle:
            logger.info(f"Model {self.model_specifier} is already loaded.")
            return True

        model_id = self.model_specifier
        logger.info(f"Loading local TTS model: {model_id} onto {self.device}")

        try:
            # Check if model is already cached (using ModelHub's logic)
            if not self.model_downloader.is_model_cached(model_id):
                logger.info(f"Model not cached locally. Downloading model {model_id}...")
                model_path = self.model_downloader.download_model(model_id)
                if not model_path:
                    raise FileNotFoundError(f"Failed to download model: {model_id}")
                logger.info(f"Model downloaded to: {model_path}")
            else:
                logger.info(f"Using cached model: {model_id}")

            # Detect model type and load
            model_id_lower = model_id.lower()
            if any(term in model_id_lower for term in self.SUPPORTED_MODELS["bark"]):
                self._load_bark_model(model_id)
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["speecht5"]):
                self._load_speecht5_model(model_id)
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["mms"]):
                self._load_vits_model(model_id)
            else:
                # Fallback: Try generic pipeline loading
                logger.info(f"Unknown model type for '{model_id}'. Attempting generic 'text-to-speech' pipeline.")
                self._load_generic_pipeline(model_id)

            logger.info(f"Model loaded successfully: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            self.unload() # Ensure cleanup on failure
            return False

    def unload(self):
        """Unload local model and free memory."""
        logger.info(f"Unloading local model: {self.model_specifier}")
        # Use helper for cleanup
        from ...utils.helpers import cleanup_resources
        cleanup_resources([self.model_handle, self.processor, self.vocoder, self.embeddings_dataset], 
                          clear_cuda=(self.device == 'cuda'))
        self.model_handle = None
        self.processor = None
        self.vocoder = None
        self.embeddings_dataset = None
        self.speaker_embeddings = None
        logger.info("Local audio model unloaded.")

    # --- Model Loading Sub-methods ---

    def _load_bark_model(self, model_id: str):
        dtype = torch.float16 if self.config.use_half_precision and self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.model_downloader.model_cache_dir)
        self.model_handle = BarkModel.from_pretrained(model_id, torch_dtype=dtype, cache_dir=self.model_downloader.model_cache_dir).to(self.device)
        self.config.sample_rate = self.model_handle.generation_config.sample_rate # Get SR from model
        logger.info(f"Bark model loaded with dtype: {dtype}, Sample Rate: {self.config.sample_rate}Hz")

    def _load_speecht5_model(self, model_id: str):
        self.processor = SpeechT5Processor.from_pretrained(model_id, cache_dir=self.model_downloader.model_cache_dir)
        self.model_handle = SpeechT5ForTextToSpeech.from_pretrained(model_id, cache_dir=self.model_downloader.model_cache_dir).to(self.device)
        try:
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir=self.model_downloader.model_cache_dir).to(self.device)
        except Exception as e:
            logger.warning(f"Could not load SpeechT5 HiFiGan vocoder: {e}. Audio quality may be lower.")
            self.vocoder = None
        self._load_speaker_embeddings() # Load embeddings for SpeechT5
        self.config.sample_rate = 16000 # SpeechT5 standard

    def _load_vits_model(self, model_id: str):
        self.processor = AutoTokenizer.from_pretrained(model_id, cache_dir=self.model_downloader.model_cache_dir)
        self.model_handle = VitsModel.from_pretrained(model_id, cache_dir=self.model_downloader.model_cache_dir).to(self.device)
        if hasattr(self.model_handle, 'config') and hasattr(self.model_handle.config, 'sampling_rate'):
            self.config.sample_rate = self.model_handle.config.sampling_rate
        else:
             logger.warning(f"Could not determine sample rate from VITS model, using config default: {self.config.sample_rate}Hz")

    def _load_generic_pipeline(self, model_id: str):
        """Loads a model using the generic 'text-to-speech' pipeline."""
        pipeline_device = 0 if self.device == "cuda" else -1
        dtype = torch.float16 if self.config.use_half_precision and self.device == "cuda" else "auto"

        self.model_handle = pipeline(
            "text-to-speech",
            model=model_id,
            device=pipeline_device,
            torch_dtype=dtype,
            cache_dir=self.model_downloader.model_cache_dir,
            trust_remote_code=True # Required for some models
        )
        self.processor = self.model_handle.tokenizer # Store tokenizer/processor if available
        
        # Try to get sample rate from pipeline/model config
        if hasattr(self.model_handle, 'model') and hasattr(self.model_handle.model, 'config'):
             model_config = self.model_handle.model.config
             sr = getattr(model_config, 'sampling_rate', getattr(model_config, 'sample_rate', None))
             if sr:
                 self.config.sample_rate = sr
                 logger.info(f"Set sample rate from pipeline config: {sr}Hz")
        
        logger.info(f"Generic TTS pipeline loaded for {model_id}")

    def _load_speaker_embeddings(self):
        """Loads the default speaker embeddings for SpeechT5."""
        if self.speaker_embeddings is not None:
            return
            
        # Use custom path if provided, else default
        embedding_source = self.config.speaker_embedding or "Matthijs/cmu-arctic-xvectors"
        
        if Path(embedding_source).is_file() and Path(embedding_source).suffix == '.npy':
            # Load from local .npy file
            try:
                embedding = np.load(embedding_source)
                self.speaker_embeddings = torch.tensor(embedding).unsqueeze(0).to(self.device)
                logger.info(f"Loaded custom speaker embedding from {embedding_source}")
                return
            except Exception as e:
                logger.error(f"Failed to load local speaker embedding '{embedding_source}': {e}. Using default.")
        
        # Download from Hugging Face datasets
        try:
            logger.info(f"Loading speaker embeddings dataset: {embedding_source}")
            self.embeddings_dataset = load_dataset(embedding_source, split="validation", cache_dir=str(CACHE_DIR / "datasets"))
            # Use a default speaker (can be made configurable later)
            default_speaker_idx = 7306 # Example index for cmu-arctic-xvectors
            if len(self.embeddings_dataset) <= default_speaker_idx:
                default_speaker_idx = 0 # Fallback to first speaker
                
            self.speaker_embeddings = torch.tensor(
                self.embeddings_dataset[default_speaker_idx]["xvector"]
            ).unsqueeze(0).to(self.device)
            logger.info(f"Using default speaker embedding (index {default_speaker_idx}) from {embedding_source}")
        except Exception as e:
            logger.error(f"Failed to load speaker embeddings: {e}. SpeechT5 may not work.", exc_info=True)
            self.speaker_embeddings = None


    # --- Generation ---

    def _generate_single_chunk(self, text: str) -> Optional[np.ndarray]:
        """Generate audio for a single text chunk using the appropriate method."""
        try:
            # Check if we're using a pipeline
            if self.model_handle and callable(self.model_handle) and not isinstance(self.model_handle, torch.nn.Module):
                return self._generate_with_pipeline(text)

            # Check for specific model types
            model_type = self.model_handle.config.model_type.lower() if hasattr(self.model_handle, 'config') else ""
            
            if "speecht5" in model_type:
                return self._generate_speecht5(text)
            elif "bark" in model_type:
                return self._generate_bark(text)
            elif "vits" in model_type or "mms" in model_type:
                return self._generate_vits(text)
            else:
                logger.error(f"Generation logic not implemented for model type: {model_type}. Attempting pipeline call.")
                # Try pipeline call as a last resort
                return self._generate_with_pipeline(text)

        except Exception as e:
             logger.error(f"Error generating audio chunk for '{text[:50]}...': {e}", exc_info=True)
             return None

    def _generate_with_pipeline(self, text: str) -> Optional[np.ndarray]:
        """Generate using the loaded Transformers pipeline."""
        if not self.model_handle: # model_handle holds the pipeline object here
            logger.error("Pipeline not loaded")
            return None
        
        try:
            logger.debug(f"Calling pipeline for text chunk: {text[:50]}...")
            output = self.model_handle(text)
            
            # Extract audio array and sample rate from pipeline output
            audio_array = None
            detected_sr = None
            
            if isinstance(output, dict):
                audio_key = 'audio' if 'audio' in output else ('waveform' if 'waveform' in output else None)
                sr_key = 'sampling_rate' if 'sampling_rate' in output else ('sample_rate' if 'sample_rate' in output else 'sr')
                
                if audio_key:
                    audio_array = output[audio_key]
                if sr_key in output:
                    detected_sr = output[sr_key]
            
            elif isinstance(output, (np.ndarray, list, torch.Tensor)):
                audio_array = output
            
            elif isinstance(output, tuple) and len(output) >= 2:
                 audio_array, detected_sr = output[0], output[1]
                 
            if audio_array is None:
                 raise GenerationError(f"Could not find 'audio' or 'waveform' in pipeline output dict: {output.keys()}", self.model_specifier)

            # Update sample rate if detected
            if detected_sr: self.config.sample_rate = int(detected_sr)

            # Convert to numpy
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            if isinstance(audio_array, list):
                audio_array = np.array(audio_array)

            # Ensure 1D float array
            audio_array = audio_array.squeeze().astype(np.float32)
            if audio_array.ndim > 1:
                 audio_array = np.mean(audio_array, axis=0) # Simple stereo to mono

            return audio_array

        except Exception as e:
            logger.error(f"Error during pipeline generation: {e}", exc_info=True)
            raise GenerationError(f"Pipeline generation failed: {e}", self.model_specifier) from e


    def _generate_speecht5(self, text: str) -> Optional[np.ndarray]:
        """Generate using SpeechT5 model."""
        if not self.model_handle or not self.processor or self.speaker_embeddings is None:
            logger.error("SpeechT5 components not fully loaded.")
            return None
            
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.vocoder:
                speech = self.model_handle.generate_speech(
                    inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
                )
            else:
                 # Generate spectrogram
                 spectrogram = self.model_handle.generate_speech(inputs["input_ids"], self.speaker_embeddings)
                 logger.warning("No vocoder loaded for SpeechT5. Returning raw spectrogram (will fail saving).")
                 # This path is problematic as it doesn't return a waveform.
                 # For a real implementation, a Griffin-Lim vocoder would be needed here as a fallback.
                 # For now, we'll let it fail downstream.
                 speech = spectrogram # This is NOT audio data
        
        # Ensure output is on CPU and numpy
        if isinstance(speech, torch.Tensor):
             speech = speech.cpu().numpy()
             
        return speech.squeeze()

    def _generate_bark(self, text: str) -> Optional[np.ndarray]:
        """Generate using Bark model."""
        if not self.model_handle or not self.processor: return None
        
        # Bark often needs voice presets, or use default
        # voice_preset = "v2/en_speaker_6" # Example
        
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        # Handle different precision
        dtype = torch.float16 if self.config.use_half_precision and self.device == "cuda" else torch.float32
        if self.model_handle.dtype != dtype:
             self.model_handle.to(dtype)
             
        with torch.no_grad():
            # generation = self.model_handle.generate(**inputs, do_sample=True, fine_temperature=0.7, coarse_temperature=0.3)
            # Bark's API changed, it might just be forward pass now
            # Let's assume pipeline usage is safer if direct call is complex
            # Update: Using generate() is standard
            generation = self.model_handle.generate(**inputs, max_new_tokens=None) # Let it run
            
        audio_array = generation.cpu().numpy().squeeze()
        return audio_array

    def _generate_vits(self, text: str) -> Optional[np.ndarray]:
        """Generate using VITS/MMS model."""
        if not self.model_handle or not self.processor: return None
        
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model_handle(**inputs).waveform
            
        return output.cpu().numpy().squeeze()
