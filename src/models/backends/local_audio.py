# llamanote/models/backends/local_audio.py
"""
Local Audio (TTS) Backend
Generates audio using local Hugging Face Transformers models.
"""

import gc
import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import AudioBackend
from ...core.types import AudioConfig, AudioResult, QuantizationConfig, LayerSplitConfig
from ...core.errors import ModelLoadError, GenerationError
from ...utils.logger import get_logger_conf, ConsoleOutput, LoggingProgress, DualProgressTracker
from ...utils.decorators import log_execution_time
from ...utils.helpers import get_device_manager, cleanup_resources
from ...utils.memory_manager import (
    CUDAMemoryManager,
    OOMRecoveryStrategy,
    with_oom_handling,
    log_memory_summary
)
from ...models.hub import ModelHub
from ...processing.audio_processor import AudioPostProcessor # Import post-processor
from ...processing.text_preprocessor import SpeakerSegmentParser # Import speaker parser
from ...models.speaker_embeddings import SpeakerEmbeddingManager, EmbeddingConfig # Import speaker manager
from ...config.manager import ConfigManager # Import ConfigManager
from ...config.settings import DEFAULT_CACHE_DIR

# Add vibevoice module to path
_VIBEVOICE_PATH = Path(__file__).parent.parent / "vibevoice"
if str(_VIBEVOICE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_VIBEVOICE_PATH.parent))

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

# --- VibeVoice components ---
_VibeVoiceForConditionalGenerationInference = None
_VibeVoiceProcessor = None
_VibeVoiceConfig = None

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

def _import_vibevoice():
    """Import VibeVoice custom components."""
    global _VibeVoiceForConditionalGenerationInference, _VibeVoiceProcessor, _VibeVoiceConfig

    if _VibeVoiceForConditionalGenerationInference is None:
        try:
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

            _VibeVoiceForConditionalGenerationInference = VibeVoiceForConditionalGenerationInference
            _VibeVoiceProcessor = VibeVoiceProcessor
            _VibeVoiceConfig = VibeVoiceConfig

            return True
        except ImportError as e:
            logger.error(f"Failed to import VibeVoice components: {e}")
            return False
    return True


logger = get_logger_conf(__name__)


# Use centralized memory management utilities
_get_gpu_memory_stats = CUDAMemoryManager.get_memory_stats
_clear_gpu_cache = CUDAMemoryManager.clear_cache


class LocalAudioBackend(AudioBackend):
    """Generate audio using local Hugging Face Transformers models."""

    SUPPORTED_MODELS = {
        "speecht5": ["microsoft/speecht5_tts"],
        "bark": ["suno/bark", "suno/bark-small"],
        "mms": ["facebook/mms-tts-eng"],
        "vits": ["facebook/mms-tts-eng"], # MMS uses VITS
        "vibevoice": ["microsoft/vibevoice-1.5b", "vibevoice/vibevoice"],
    }

    def __init__(self, config: AudioConfig, model_specifier: str, layer_split_config: Optional[LayerSplitConfig] = None):
        super().__init__("local_audio", model_specifier, config)
        _import_transformers() # Ensure transformers is available

        self.model = None
        self.processor = None
        self.vocoder = None # Specific to SpeechT5
        self.pipeline = None # For pipeline-based models
        self.device = get_device_manager().get_device()

        # Memory optimization configuration
        self.layer_split_config = layer_split_config or LayerSplitConfig()

        self.config_manager = ConfigManager()
        self.model_cache_dir = self.config_manager.get_dir("model_cache")

        self.model_hub = ModelHub(cache_dir=self.model_cache_dir)
        self.speaker_embeddings = None
        self.embeddings_dataset = None
        self.post_processor = AudioPostProcessor() # Use the separated class

        # Multi-speaker support (Phase 2)
        self.speaker_parser = SpeakerSegmentParser()
        self.speaker_manager: Optional[SpeakerEmbeddingManager] = None
        self._init_speaker_manager()

    @property
    def provider_identifier(self) -> str:
        return "local_audio"

    def _init_speaker_manager(self):
        """Initialize the speaker embedding manager based on config."""
        if not self.config.enable_multi_speaker:
            self.logger.debug("Multi-speaker support disabled")
            return

        try:
            # Create embedding config from AudioConfig
            embedding_config = EmbeddingConfig(
                embedding_dim=512,  # SpeechT5 standard
                normalize=True,
                seed=self.config.speaker_embedding_seed,
                distribution=self.config.speaker_random_distribution,
                dataset_name=self.config.speaker_dataset_name,
                gender_filter=self.config.default_speaker_gender if self.config.default_speaker_gender != "neutral" else None
            )

            # Create speaker manager
            cache_dir = self.config.speaker_embedding_cache_dir or (self.model_cache_dir / "speakers")
            self.speaker_manager = SpeakerEmbeddingManager(
                method=self.config.speaker_embedding_method,
                config=embedding_config,
                cache_dir=cache_dir,
                enable_cache=True,
                enable_persistence=True
            )

            self.logger.info(f"Speaker embedding manager initialized: method={self.config.speaker_embedding_method}")

        except Exception as e:
            self.logger.error(f"Failed to initialize speaker manager: {e}", exc_info=True)
            self.speaker_manager = None

    @log_execution_time(logger_name=__name__)
    def load(self, **kwargs) -> bool:
        """Load local TTS model from Hugging Face."""
        if self.model_handle is not None:
            self.logger.info("Local audio model already loaded.")
            return True

        model_id = self.model_specifier
        self.logger.info(f"Loading local TTS model: {model_id} onto {self.device}")
        ConsoleOutput.info(f"Loading local audio model: {model_id}...")

        # --- Display Memory Projection for Audio Model (if enabled) ---
        try:
            from ...core.types import LayerSplitConfig
            from ...utils.memory_estimator import display_memory_projection

            # Check if show_memory_projection is enabled (from config or kwargs)
            show_projection = kwargs.get('show_memory_projection', True)  # Default to True for audio

            if show_projection:
                self.logger.info(f"Displaying memory projection for audio model: {model_id}")
                projection = display_memory_projection(
                    model_id=model_id,
                    quantization=self.config.quantization or "none",
                    max_seq_length=512,  # Audio models typically use shorter sequences
                    cache_dir=self.model_cache_dir,
                    trust_remote_code=True
                )
                self.logger.info(f"Audio model memory projection: Total={projection.total_size_mb:.0f}MB")
        except Exception as e:
            self.logger.debug(f"Could not display audio model memory projection: {e}")

        try:
            # Check if model is already cached/downloaded
            if not self.model_hub.is_model_cached(model_id): # This checks registry, might need to check hub cache path
                self.logger.info(f"Model not cached, downloading: {model_id}")
                # Note: model_hub.download_model uses snapshot_download
                model_path = self.model_hub.download_model(model_id)
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
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["vibevoice"]):
                self._load_vibevoice_model(model_id)
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
        """Unload local model and free memory with aggressive cleanup."""
        self.logger.info(f"Unloading local audio model: {self.model_specifier}")

        # Log memory before unload
        if self.device == "cuda":
            mem_stats = _get_gpu_memory_stats()
            self.logger.info(f"GPU memory before unload: {mem_stats.get('allocated_mb', 0):.1f}MB allocated")

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

        # Aggressive memory cleanup
        _clear_gpu_cache(aggressive=True)

        # Log memory after unload
        if self.device == "cuda":
            mem_stats = _get_gpu_memory_stats()
            self.logger.info(f"GPU memory after unload: {mem_stats.get('allocated_mb', 0):.1f}MB allocated")

        self.logger.info("Local audio model unloaded.")

    def _load_bark_model(self, model_id: str):
        self.processor = _AutoProcessor.from_pretrained(model_id, cache_dir=self.model_cache_dir)
        dtype = torch.float16 if self.config.use_half_precision and self.device == "cuda" else torch.float32
        self.model = _BarkModel.from_pretrained(model_id, dtype=dtype, cache_dir=self.model_cache_dir).to(self.device)
        self.config.sample_rate = self.model.generation_config.sample_rate # Get SR from model
        self.logger.info(f"Bark model loaded with dtype: {dtype}, Sample Rate: {self.config.sample_rate}Hz")

    def _load_speecht5_model(self, model_id: str):
        load_dataset = _import_datasets() # Ensure datasets is available

        self.processor = _SpeechT5Processor.from_pretrained(model_id, cache_dir=self.model_cache_dir)
        self.model = _SpeechT5ForTextToSpeech.from_pretrained(model_id, cache_dir=self.model_cache_dir).to(self.device)

        # Vocoder is essential for SpeechT5
        try:
            vocoder_id = "microsoft/speecht5_hifigan"
            self.vocoder = _SpeechT5HifiGan.from_pretrained(vocoder_id, cache_dir=self.model_cache_dir).to(self.device)
        except Exception as e:
            self.logger.warning(f"Could not load SpeechT5 HiFiGan vocoder: {e}. Audio quality may be low.")
            self.vocoder = None

        # Load speaker embeddings
        if self.config.speaker_embedding:
            self.logger.warning("Custom speaker embeddings not yet supported, using default.")
            # TODO: Add logic to load custom embedding path/ID

        if self.speaker_embeddings is None:
             self.logger.info("Loading default speaker embeddings (cmu-arctic-xvectors)...")
             try:
                 # Try loading without trust_remote_code first (for converted datasets)
                 self.embeddings_dataset = load_dataset(
                     "Matthijs/cmu-arctic-xvectors",
                     split="validation",
                     cache_dir=self.model_cache_dir / "datasets"
                 )
                 # Use a common, generic-sounding speaker
                 default_speaker_idx = 7306 # Example index
                 self.speaker_embeddings = torch.tensor(
                     self.embeddings_dataset[default_speaker_idx]["xvector"]
                 ).unsqueeze(0).to(self.device)
                 self.logger.info(f"Using default speaker embedding index: {default_speaker_idx}")
             except Exception as e:
                 self.logger.warning(f"Could not load embeddings dataset: {e}. Trying alternative method...")
                 # Fallback: Use a bundled default embedding
                 try:
                     self._load_default_speaker_embedding()
                 except Exception as e2:
                     self.logger.error(f"Failed to load speaker embeddings: {e2}. SpeechT5 may not work.", exc_info=True)
                     raise ModelLoadError("Failed to load SpeechT5 speaker embeddings", model_id) from e2

        self.config.sample_rate = 16000 # SpeechT5 standard

    def _load_default_speaker_embedding(self):
        """Load a default speaker embedding as fallback when dataset loading fails."""
        from huggingface_hub import hf_hub_download
        import json

        self.logger.info("Loading default speaker embedding from HuggingFace Hub...")

        try:
            # Try to download parquet file from the auto-converted dataset
            parquet_file = hf_hub_download(
                repo_id="Matthijs/cmu-arctic-xvectors",
                filename="data/validation-00000-of-00001.parquet",
                repo_type="dataset",
                cache_dir=self.model_cache_dir / "datasets"
            )

            # Load parquet using pandas
            import pandas as pd
            df = pd.read_parquet(parquet_file)

            # Get a speaker embedding (use index 7306 like before, or first if not available)
            if len(df) > 7306:
                embedding_data = df.iloc[7306]['xvector']
            else:
                embedding_data = df.iloc[0]['xvector']

            # Convert to numpy array if needed
            if isinstance(embedding_data, list):
                embedding_array = np.array(embedding_data, dtype=np.float32)
            else:
                embedding_array = np.array(embedding_data, dtype=np.float32)

            self.speaker_embeddings = torch.tensor(embedding_array).unsqueeze(0).to(self.device)
            self.logger.info("Loaded default speaker embedding from parquet file.")

        except Exception as e:
            self.logger.warning(f"Could not load from parquet file: {e}. Using hardcoded embedding.")
            # Fallback: Use a zero/random embedding (this may produce poor quality audio)
            # A proper embedding should be 512-dimensional for SpeechT5
            default_embedding = np.random.randn(512).astype(np.float32) * 0.1
            self.speaker_embeddings = torch.tensor(default_embedding).unsqueeze(0).to(self.device)
            self.logger.warning("Using random speaker embedding. Audio quality may be degraded.")

    def _load_vits_model(self, model_id: str):
        # Covers MMS and potentially other VITS models
        self.processor = _AutoTokenizer.from_pretrained(model_id, cache_dir=self.model_cache_dir)
        self.model = _VitsModel.from_pretrained(model_id, cache_dir=self.model_cache_dir).to(self.device)
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'sampling_rate'):
            self.config.sample_rate = self.model.config.sampling_rate
        else:
             self.config.sample_rate = 22050 # Common VITS default
        self.logger.info(f"VITS/MMS model loaded. Sample Rate: {self.config.sample_rate}Hz")

    def _load_vibevoice_model(self, model_id: str):
        """Load VibeVoice model with processor and advanced memory optimizations including OOM handling."""
        self.logger.info(f"Loading VibeVoice model: {model_id}")

        # Import VibeVoice components
        if not _import_vibevoice():
            raise ModelLoadError("Failed to import VibeVoice components", model_id)

        try:
            # Clear cache before loading
            _clear_gpu_cache(aggressive=True)

            # Load processor first (lightweight)
            self.logger.info("Loading VibeVoice processor...")
            self.processor = _VibeVoiceProcessor.from_pretrained(
                model_id,
                cache_dir=self.model_cache_dir
            )

            # Configure quantization for memory efficiency
            quantization_config = QuantizationConfig(method=self.config.quantization)
            bnb_config = quantization_config.to_bnb_config()

            # Configure dtype based on device and user settings
            if self.device == "cuda" and self.config.use_half_precision:
                dtype = torch.bfloat16  # VibeVoice works better with bfloat16
            else:
                dtype = torch.float32

            # Prepare offload folder for disk offloading
            offload_folder = None
            if self.config.enable_disk_offload and self.layer_split_config.offload_folder:
                offload_folder = str(self.layer_split_config.offload_folder)
                self.logger.info(f"Disk offloading enabled to: {offload_folder}")

            # Log GPU memory before loading
            if self.device == "cuda":
                log_memory_summary("before VibeVoice loading", self.logger)

            # Define model loading function for OOM handler
            def load_model_fn(**strategy_params):
                """Model loading function that can be retried with different parameters."""
                # Merge strategy params with base config
                device_map = strategy_params.get('device_map', 'auto' if self.layer_split_config.enabled and self.device == "cuda" else None)
                max_memory = strategy_params.get('max_memory', self.layer_split_config.get_max_memory_dict() if self.layer_split_config.enabled else None)

                # Override with strategy params if provided
                low_cpu_mem = strategy_params.get('low_cpu_mem_usage', self.layer_split_config.low_cpu_mem_usage)
                offload_state = strategy_params.get('offload_state_dict', self.layer_split_config.offload_state_dict)

                self.logger.info("Loading VibeVoice model with memory optimizations...")
                self.logger.info(f"  - Quantization: {self.config.quantization}")
                self.logger.info(f"  - Dtype: {dtype}")
                self.logger.info(f"  - Device map: {device_map}")
                self.logger.info(f"  - Max memory: {max_memory}")
                self.logger.info(f"  - Offload folder: {offload_folder}")
                self.logger.info(f"  - Low CPU mem: {low_cpu_mem}")
                self.logger.info(f"  - Offload state dict: {offload_state}")

                # Build kwargs for from_pretrained
                load_kwargs = {
                    'cache_dir': self.model_cache_dir,
                    'quantization_config': bnb_config,
                    'dtype': dtype,
                    'attn_implementation': "eager",
                }

                # Only add device_map and memory params if using offloading
                if device_map:
                    load_kwargs['device_map'] = device_map
                if max_memory:
                    load_kwargs['max_memory'] = max_memory
                if low_cpu_mem and device_map:
                    load_kwargs['low_cpu_mem_usage'] = low_cpu_mem
                if offload_folder and device_map:
                    load_kwargs['offload_folder'] = offload_folder
                if offload_state and device_map:
                    load_kwargs['offload_state_dict'] = offload_state

                self.logger.debug(f"from_pretrained kwargs: {load_kwargs}")

                try:
                    model = _VibeVoiceForConditionalGenerationInference.from_pretrained(
                        model_id,
                        **load_kwargs
                    )
                    return model
                except Exception as e:
                    self.logger.error(f"Exception during from_pretrained: {type(e).__name__}: {e}")
                    raise

            # Try to load the model
            # First attempt without OOM handling to see if it works
            try:
                self.logger.info("Attempting initial model load...")
                self.model = load_model_fn()
                self.logger.info("✓ Model loaded successfully on first attempt")
            except Exception as e:
                # Check if OOM and if OOM handling is enabled
                if CUDAMemoryManager.is_oom_error(e) and self.layer_split_config.auto_oom_handling and self.device == "cuda":
                    self.logger.warning(f"Initial load failed with OOM error: {e}")
                    self.logger.info("🛡️  Activating automatic OOM recovery")

                    # Create OOM recovery strategy
                    initial_gpu_mem = list(self.layer_split_config.max_gpu_memory.values())[0] if self.layer_split_config.max_gpu_memory else "4GB"
                    recovery_strategy = OOMRecoveryStrategy(
                        initial_gpu_memory=initial_gpu_mem,
                        initial_cpu_memory=self.layer_split_config.max_cpu_memory,
                        min_gpu_memory="1GB",
                        offload_folder=self.layer_split_config.offload_folder,  # Pass offload folder
                        logger=self.logger
                    )

                    # Load with OOM handling
                    try:
                        self.model, final_params = with_oom_handling(
                            load_fn=load_model_fn,
                            recovery_strategy=recovery_strategy,
                            logger=self.logger
                        )
                        if final_params:
                            self.logger.info(f"✓ Model loaded with adjusted parameters: {final_params}")
                    except Exception as recovery_error:
                        if CUDAMemoryManager.is_oom_error(recovery_error):
                            self.logger.error("Failed to load model: CUDA Out of Memory after all recovery attempts")
                            self.logger.error("Consider: 1) Enabling disk offloading, 2) Using smaller model, 3) Reducing context size")
                        raise
                else:
                    # Not an OOM error or OOM handling disabled - re-raise original error
                    self.logger.error(f"Model loading failed: {type(e).__name__}: {e}")
                    raise

            # Log GPU memory after loading
            if self.device == "cuda":
                log_memory_summary("after VibeVoice loading", self.logger)

            # Get sample rate from config
            if hasattr(self.processor, 'audio_processor') and hasattr(self.processor.audio_processor, 'sampling_rate'):
                self.config.sample_rate = self.processor.audio_processor.sampling_rate
            else:
                self.config.sample_rate = 24000  # VibeVoice default is 24kHz

            # Log successful loading with configuration summary
            self.logger.info(f"✓ VibeVoice model loaded successfully")
            self.logger.info(f"  - Sample Rate: {self.config.sample_rate}Hz")
            self.logger.info(f"  - Dtype: {dtype}")
            self.logger.info(f"  - CPU offloading: {'Enabled' if hasattr(self.model, 'hf_device_map') else 'Disabled'}")
            if hasattr(self.model, 'hf_device_map'):
                self.logger.info(f"  - Device distribution: {self.model.hf_device_map}")

            # Final cache clear
            _clear_gpu_cache()

        except Exception as e:
            self.logger.error(f"Failed to load VibeVoice model: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load VibeVoice: {e}", model_id) from e


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
                cache_dir=self.model_cache_dir
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
                      checkpoint_callback: Optional[Any] = None,
                      checkpoint_interval: int = 5,
                      resume_from_chunk: Optional[int] = None,
                      dual_tracker: Optional[DualProgressTracker] = None,
                      **kwargs) -> Optional[AudioResult]:
        """
        Generate audio from text using the loaded local model with checkpointing support.

        Args:
            text: Text to convert to audio
            output_path: Path to save audio file
            chunk_text: Whether to split long text into chunks
            checkpoint_callback: Optional callback function to save checkpoint
            checkpoint_interval: Save checkpoint every N chunks
            resume_from_chunk: If provided, resume from this chunk index
            dual_tracker: Optional dual progress tracker for displaying progress
            **kwargs: Additional arguments

        Returns:
            AudioResult object or None if generation failed
        """
        if self.model_handle is None:
            self.logger.error("No local model or pipeline loaded.")
            return None

        output_path = self._resolve_output_path(output_path)
        self.logger.info(f"Generating local audio for {len(text)} characters to {output_path.name}")
        start_time = time.time()

        try:
            # Check for multi-speaker content
            if self.config.enable_multi_speaker and self.speaker_manager:
                has_speakers = self.speaker_parser.has_multiple_speakers(text)
                if has_speakers:
                    self.logger.info("Multi-speaker content detected, using speaker-specific voices")
                    return self._generate_multispeaker_audio(
                        text, output_path, checkpoint_callback, checkpoint_interval,
                        resume_from_chunk, dual_tracker, **kwargs
                    )

            # Single speaker or multi-speaker disabled - use standard generation
            # Split text into chunks if needed
            audio_arrays = []
            text_chunks = []

            if chunk_text and len(text) > self.config.chunk_size:
                text_chunks = self._split_text(text)
                self.logger.info(f"Splitting text into {len(text_chunks)} chunks.")
            else:
                text_chunks = [text]

            # Determine starting point
            start_index = resume_from_chunk if resume_from_chunk is not None else 0
            if start_index > 0:
                self.logger.info(f"Resuming audio generation from chunk {start_index}/{len(text_chunks)}")
                # For audio, we need placeholder arrays (these should be loaded from checkpoint)
                for i in range(start_index):
                    # Add empty placeholder - actual data should come from checkpoint
                    audio_arrays.append(np.array([], dtype=np.float32))

            # Initialize dual tracker stage progress BEFORE entering context
            if dual_tracker:
                dual_tracker.set_stage_progress(start_index, len(text_chunks), "Generating audio chunks")

            with LoggingProgress(self.logger, "Generating audio chunks", len(text_chunks), dual_tracker=dual_tracker) as progress:
                # Update progress for skipped chunks
                for _ in range(start_index):
                    progress.update(1)

                for i in range(start_index, len(text_chunks)):
                    chunk = text_chunks[i]
                    audio_chunk = self._generate_single_chunk(chunk)
                    if audio_chunk is not None and audio_chunk.size > 0:
                        audio_arrays.append(audio_chunk)
                    progress.update(1)

                    # Save checkpoint if callback provided and interval reached
                    if checkpoint_callback and checkpoint_interval > 0:
                        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(text_chunks):
                            self.logger.debug(f"Saving audio checkpoint at chunk {i + 1}/{len(text_chunks)}")

                            # NOTE: DO NOT save intermediate audio files - they cause massive disk bloat
                            # Only save checkpoint metadata without generating cumulative audio files
                            # The checkpoint system stores audio_arrays in compressed format already

                            # Save checkpoint with current audio arrays (no intermediate file)
                            checkpoint_callback(i + 1, audio_arrays, {
                                "text_chunks": text_chunks,
                                "sample_rate": self.config.sample_rate,
                                "intermediate_audio_path": None  # Don't save intermediate files
                            })

            if not audio_arrays:
                 raise GenerationError("No valid audio generated from any chunks.", self.model_specifier)

            # Filter out empty placeholder arrays from resume
            audio_arrays = [arr for arr in audio_arrays if arr.size > 0]

            if not audio_arrays:
                raise GenerationError("No valid audio after filtering placeholders.", self.model_specifier)

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

    def _generate_multispeaker_audio(
        self,
        text: str,
        output_path: Path,
        checkpoint_callback: Optional[Any] = None,
        checkpoint_interval: int = 5,
        resume_from_chunk: Optional[int] = None,
        dual_tracker: Optional[DualProgressTracker] = None,
        **kwargs
    ) -> Optional[AudioResult]:
        """
        Generate audio with multiple speakers using distinct voices.

        Parses speaker segments from text and generates audio for each segment
        with the appropriate speaker embedding.
        """
        start_time = time.time()

        try:
            # Parse speaker segments
            segments = self.speaker_parser.parse_speakers(text)
            if not segments:
                self.logger.warning("No speaker segments found, falling back to single-speaker mode")
                return self._generate_single_speaker_fallback(text, output_path, **kwargs)

            unique_speakers = self.speaker_parser.get_unique_speakers(text)
            self.logger.info(f"Found {len(segments)} segments from {len(unique_speakers)} speakers: {unique_speakers}")

            # Get embeddings for all speakers
            speaker_embeddings = {}
            for speaker in unique_speakers:
                embedding = self.speaker_manager.get_embedding(speaker, device=str(self.device))
                speaker_embeddings[speaker] = embedding
                self.logger.debug(f"Generated embedding for speaker '{speaker}'")

            # Generate audio for each segment
            audio_arrays = []
            start_index = resume_from_chunk if resume_from_chunk is not None else 0

            if dual_tracker:
                dual_tracker.set_stage_progress(start_index, len(segments), "Generating multi-speaker audio")

            with LoggingProgress(self.logger, "Generating speaker segments", len(segments), dual_tracker=dual_tracker) as progress:
                # Update progress for skipped chunks
                for _ in range(start_index):
                    progress.update(1)

                for i in range(start_index, len(segments)):
                    segment = segments[i]
                    speaker_name = segment['speaker']
                    segment_text = segment['text']

                    # Get speaker embedding
                    speaker_emb = speaker_embeddings.get(speaker_name)
                    if speaker_emb is None:
                        self.logger.warning(f"No embedding for speaker '{speaker_name}', using default")
                        speaker_emb = self.speaker_embeddings

                    # Temporarily swap speaker embedding
                    original_embedding = self.speaker_embeddings
                    self.speaker_embeddings = speaker_emb.unsqueeze(0) if len(speaker_emb.shape) == 1 else speaker_emb

                    # Generate audio for this segment
                    audio_chunk = self._generate_single_chunk(segment_text)

                    # Restore original embedding
                    self.speaker_embeddings = original_embedding

                    if audio_chunk is not None and audio_chunk.size > 0:
                        audio_arrays.append(audio_chunk)
                    else:
                        self.logger.warning(f"Failed to generate audio for segment {i} (speaker: {speaker_name})")

                    progress.update(1)

                    # Checkpoint saving
                    if checkpoint_callback and checkpoint_interval > 0:
                        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(segments):
                            self.logger.debug(f"Saving checkpoint at segment {i + 1}/{len(segments)}")

                            # NOTE: DO NOT save intermediate audio files - they cause massive disk bloat
                            # Only save checkpoint with audio_arrays (compressed by checkpoint system)
                            try:
                                checkpoint_callback(i + 1, audio_arrays, {
                                    "segments": segments,
                                    "sample_rate": self.config.sample_rate,
                                    "intermediate_audio_path": None,  # Don't save intermediate files
                                    "total_segments": len(segments)
                                })
                            except Exception as checkpoint_err:
                                self.logger.warning(f"Failed to save checkpoint: {checkpoint_err}")

            # Combine all audio segments
            if not audio_arrays:
                self.logger.error("No audio segments were generated")
                return None

            valid_arrays = [arr for arr in audio_arrays if arr.size > 0]
            if not valid_arrays:
                self.logger.error("All audio segments are empty")
                return None

            combined_audio = self._combine_audio(valid_arrays, self.config.sample_rate)
            if combined_audio is None or combined_audio.size == 0:
                self.logger.error("Failed to combine audio segments")
                return None

            # Post-process and save
            processed_audio = self.post_processor._post_process_audio(
                combined_audio,
                self.config.sample_rate,
                self.config.speed,
                self.config.pitch_shift,
                self.config.volume_normalize
            )

            saved_path = self.post_processor._save_audio(
                output_path,
                processed_audio,
                self.config.sample_rate,
                self.config.output_format
            )

            if not saved_path:
                return None

            # Create result
            duration = len(processed_audio) / self.config.sample_rate
            processing_time = time.time() - start_time

            result = AudioResult(
                audio_path=saved_path,
                duration_seconds=duration,
                sample_rate=self.config.sample_rate,
                num_samples=len(processed_audio),
                model_used=f"{self.model_identifier} (multi-speaker: {len(unique_speakers)} voices)",
                processing_time=processing_time,
                chunks_processed=len(segments)
            )

            self.logger.info(
                f"Multi-speaker audio generated: {result.duration_formatted} duration, "
                f"{len(unique_speakers)} speakers, {len(segments)} segments"
            )

            # Save speaker mappings
            if self.speaker_manager:
                self.speaker_manager.save_mappings()

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate multi-speaker audio: {e}", exc_info=True)
            return None

    def _generate_single_speaker_fallback(
        self,
        text: str,
        output_path: Path,
        **kwargs
    ) -> Optional[AudioResult]:
        """Fallback to single-speaker generation if multi-speaker parsing fails."""
        self.logger.info("Using single-speaker fallback")
        # Just call the regular generation logic
        text_chunks = self._split_text(text) if len(text) > self.config.chunk_size else [text]

        audio_arrays = []
        for chunk in text_chunks:
            audio_chunk = self._generate_single_chunk(chunk)
            if audio_chunk is not None and audio_chunk.size > 0:
                audio_arrays.append(audio_chunk)

        if not audio_arrays:
            return None

        combined = self._combine_audio(audio_arrays, self.config.sample_rate)
        processed = self.post_processor._post_process_audio(
            combined, self.config.sample_rate,
            self.config.speed, self.config.pitch_shift, self.config.volume_normalize
        )

        saved_path = self.post_processor._save_audio(
            output_path, processed, self.config.sample_rate, self.config.output_format
        )

        if saved_path:
            duration = len(processed) / self.config.sample_rate
            return AudioResult(
                audio_path=saved_path,
                duration_seconds=duration,
                sample_rate=self.config.sample_rate,
                num_samples=len(processed),
                model_used=self.model_identifier,
                processing_time=0,
                chunks_processed=len(audio_arrays)
            )
        return None

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
            elif any(term in model_id_lower for term in self.SUPPORTED_MODELS["vibevoice"]):
                return self._generate_vibevoice(text)
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

        # SpeechT5 has a max sequence length of 600 tokens
        # Check if text is too long and split if necessary
        MAX_TOKENS = 590  # Use slightly less than 600 for safety
        inputs = self.processor(text=text, return_tensors="pt")

        if inputs["input_ids"].shape[1] > MAX_TOKENS:
            self.logger.warning(f"Text chunk has {inputs['input_ids'].shape[1]} tokens, exceeding SpeechT5 limit of {MAX_TOKENS}. Splitting further...")

            # Split the text into smaller pieces
            words = text.split()
            mid = len(words) // 2
            first_half = ' '.join(words[:mid])
            second_half = ' '.join(words[mid:])

            # Recursively generate for each half
            audio1 = self._generate_speecht5(first_half)
            audio2 = self._generate_speecht5(second_half)

            if audio1 is not None and audio2 is not None:
                # Combine the two halves
                return np.concatenate([audio1, audio2])
            elif audio1 is not None:
                return audio1
            elif audio2 is not None:
                return audio2
            else:
                return None

        # Text is within limits, generate normally
        inputs = inputs.to(self.device)
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
        # Convert to float32 if needed (NumPy doesn't support bfloat16)
        return speech.cpu().float().numpy().squeeze()

    def _generate_bark(self, text: str) -> Optional[np.ndarray]:
        if not self.model or not self.processor: return None
        # Bark generation often needs presets (e.g., "v2/en_speaker_6")
        voice_preset = "v2/en_speaker_6" # Make this configurable?
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
        
        # Bark output is usually 24kHz
        self.config.sample_rate = self.model.generation_config.sample_rate
        # Convert to float32 if needed (NumPy doesn't support bfloat16)
        return audio_array.cpu().float().numpy().squeeze()

    def _generate_vits(self, text: str) -> Optional[np.ndarray]:
        if not self.model or not self.processor: return None
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            waveform = output.waveform if hasattr(output, 'waveform') else output[0]

        # SR should be in config from loading
        # Convert to float32 if needed (NumPy doesn't support bfloat16)
        return waveform.cpu().float().numpy().squeeze()

    def _generate_vibevoice(self, text: str) -> Optional[np.ndarray]:
        """Generate audio using VibeVoice model with memory management and OOM recovery."""
        if not self.model or not self.processor:
            self.logger.error("VibeVoice components not fully loaded.")
            return None

        # Clear cache before generation if enabled
        if self.config.clear_cache_between_chunks:
            _clear_gpu_cache()

        # Process input text with VibeVoice processor
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device (might be CPU if offloaded)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Progressive generation parameters for OOM recovery
        generation_params = {
            'max_new_tokens': 2048,
            'inference_steps': 10,
            'cfg_scale': 1.3
        }

        # Try generation with progressive parameter reduction on OOM
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Log generation parameters
                if attempt > 0:
                    self.logger.info(f"Generation attempt {attempt + 1}/{max_attempts} with reduced parameters:")
                    self.logger.info(f"  - max_new_tokens: {generation_params['max_new_tokens']}")
                    self.logger.info(f"  - inference_steps: {generation_params['inference_steps']}")

                # Clear cache before each attempt
                if attempt > 0:
                    _clear_gpu_cache(aggressive=True)

                # Generate audio with the model
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        max_new_tokens=generation_params['max_new_tokens'],
                        temperature=0.7,
                        do_sample=True,
                        cfg_scale=generation_params['cfg_scale'],
                        inference_steps=generation_params['inference_steps']
                    )

                # Extract audio array from output
                # VibeVoice returns audio in speech_outputs
                if hasattr(output, 'speech_outputs') and output.speech_outputs:
                    # Concatenate all speech outputs if multiple
                    audio_arrays = []
                    for speech_output in output.speech_outputs:
                        if isinstance(speech_output, torch.Tensor):
                            # Convert to float32 if needed (NumPy doesn't support bfloat16)
                            arr = speech_output.cpu().float().numpy().squeeze()
                            audio_arrays.append(arr)
                        else:
                            arr = np.array(speech_output).squeeze()
                            audio_arrays.append(arr)

                    if len(audio_arrays) == 1:
                        audio_array = audio_arrays[0]
                    else:
                        # Flatten each array to 1D and concatenate
                        audio_arrays = [arr.flatten() for arr in audio_arrays]
                        audio_array = np.concatenate(audio_arrays, axis=0)

                    # Clear intermediate tensors and cache after generation
                    del output
                    if self.config.clear_cache_between_chunks:
                        _clear_gpu_cache()

                    if attempt > 0:
                        self.logger.info(f"✓ Generation succeeded with reduced parameters on attempt {attempt + 1}")

                    return audio_array.squeeze()
                else:
                    self.logger.error(f"Unexpected VibeVoice output format: {type(output)}")
                    return None

            except Exception as e:
                # Check if this is an OOM error
                if CUDAMemoryManager.is_oom_error(e):
                    mem_stats = _get_gpu_memory_stats()
                    self.logger.warning(
                        f"OOM during generation (attempt {attempt + 1}/{max_attempts}): "
                        f"GPU {mem_stats.get('allocated_mb', 0):.0f}MB allocated, "
                        f"{mem_stats.get('free_mb', 0):.0f}MB free"
                    )

                    # If not the last attempt, reduce parameters and retry
                    if attempt < max_attempts - 1:
                        # Progressive reduction strategy
                        if attempt == 0:
                            # First retry: Reduce inference steps
                            generation_params['inference_steps'] = max(5, generation_params['inference_steps'] // 2)
                            self.logger.info(f"Reducing inference_steps to {generation_params['inference_steps']}")
                        elif attempt == 1:
                            # Second retry: Further reduce inference steps and max tokens
                            generation_params['inference_steps'] = 3
                            generation_params['max_new_tokens'] = generation_params['max_new_tokens'] // 2
                            self.logger.info(f"Reducing max_new_tokens to {generation_params['max_new_tokens']}, inference_steps to {generation_params['inference_steps']}")
                        elif attempt == 2:
                            # Third retry: Minimal settings
                            generation_params['max_new_tokens'] = 512
                            generation_params['inference_steps'] = 2
                            generation_params['cfg_scale'] = 1.0
                            self.logger.info("Using minimal generation settings")
                        else:
                            # Fourth retry: Absolute minimum
                            generation_params['max_new_tokens'] = 256
                            generation_params['inference_steps'] = 1
                            generation_params['cfg_scale'] = 1.0
                            self.logger.info("Using absolute minimum settings")

                        # Continue to next attempt
                        continue
                    else:
                        # Last attempt failed - log and return None
                        self.logger.error(f"OOM error persists after {max_attempts} attempts with reduced parameters")
                        self.logger.error("Text chunk may be too long. Consider reducing chunk_size in audio config.")
                        return None
                else:
                    # Not an OOM error - log and return None
                    self.logger.error(f"Error during VibeVoice generation: {e}", exc_info=True)
                    return None

        # If we exhausted all attempts without success
        self.logger.error(f"Failed to generate audio after {max_attempts} attempts")
        return None

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
