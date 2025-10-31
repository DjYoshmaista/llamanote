# llamanote/models/backends/openai_audio.py
"""
OpenAI TTS (Text-to-Speech) Backend
Generates audio using the OpenAI TTS API.
"""

import time
from pathlib import Path
from typing import Optional, Dict

from .base import AudioBackend
from ...core.types import AudioConfig, AudioResult
from ...core.errors import ModelLoadError, GenerationError
from ...utils.logger import get_logger_conf, ConsoleOutput, LoggingProgress
from ...utils.decorators import log_execution_time
from ...processing.audio_processor import AudioPostProcessor # Import post-processor
from ...config.settings import DEFAULT_CACHE_DIR

# Try importing OpenAI library
try:
    import openai
    from openai import OpenAI, AuthenticationError
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OpenAI = None # type: ignore
    AuthenticationError = None # type: ignore
    OPENAI_AVAILABLE = False

logger = get_logger_conf(__name__)

class OpenAIAudioBackend(AudioBackend):
    """Generate audio using OpenAI's Text-to-Speech API."""

    def __init__(self,
                 config: AudioConfig,
                 model_specifier: str, # e.g., "tts-1", "tts-1-hd"
                 api_key: str):
        super().__init__("openai_audio", model_specifier, config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run 'pip install openai'")
            
        self.api_key = api_key
        # self.model_handle (from base) will hold the client
        self.post_processor = AudioPostProcessor() # Use the separated class

        if model_specifier not in ["tts-1", "tts-1-hd"]:
             self.logger.warning(f"Unsupported OpenAI model '{model_specifier}', defaulting to 'tts-1'.")
             self.model_specifier = "tts-1"

    @property
    def provider_identifier(self) -> str:
        return "openai_audio"

    def load(self, **kwargs) -> bool:
        """Initialize the OpenAI client and test the API key."""
        if self.model_handle:
            return True
            
        try:
            self.model_handle = OpenAI(api_key=self.api_key)
            self.model_handle.models.list() # Test API key
            self.logger.info("OpenAI API client initialized and key verified (for audio).")
            return True
        except AuthenticationError as e:
             logger.error(f"OpenAI API key is invalid: {e}")
             ConsoleOutput.error("OpenAI API key is invalid. Please check your configuration.")
             raise ModelLoadError(f"OpenAI API key invalid: {e}", self.model_specifier) from e
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client or verify key: {e}", exc_info=True)
            ConsoleOutput.error(f"OpenAI connection failed: {e}. Check key and network access.")
            raise ModelLoadError(f"OpenAI connection failed: {e}", self.model_specifier) from e

    def unload(self):
        """No explicit unloading needed, just clear client reference."""
        if self.model_handle is not None:
             del self.model_handle
             self.model_handle = None
             logger.debug("OpenAI (audio) client reference cleared.")
        pass

    @log_execution_time(logger_name=__name__)
    def generate_audio(self,
                      text: str,
                      output_path: Optional[Path] = None,
                      chunk_text: bool = True, # OpenAI API handles chunks up to 4096 chars
                      **kwargs) -> Optional[AudioResult]:
        """Generate audio using OpenAI API."""
        if self.model_handle is None:
            self.logger.error("OpenAI client not loaded.")
            return None
            
        output_path = self._resolve_output_path(output_path)
        
        # OpenAI TTS API supports 'mp3', 'opus', 'aac', 'flac'.
        # 'wav' is not directly supported, so we must generate 'flac' (lossless) or 'mp3'
        # and then convert if 'wav' is requested.
        api_format = self.config.output_format.lower()
        if api_format not in ["mp3", "opus", "aac", "flac"]:
             self.logger.info(f"Target format {api_format} not supported by OpenAI. Generating FLAC and will convert.")
             api_format = "flac" # Use flac for best quality before conversion
             
        # Temporary path for the API output if conversion or post-processing is needed
        needs_post_processing = (
            self.config.pitch_shift != 0 or
            self.config.volume_normalize or
            api_format != self.config.output_format.lower() # Format conversion needed
        )
        
        temp_output_path = output_path.with_suffix(f".{api_format}") if needs_post_processing else output_path
        final_output_path = output_path # The path the user requested (with correct extension)
        
        # OpenAI API has a limit of 4096 characters.
        # We must split the text if it's longer.
        text_chunks = [text]
        if chunk_text and len(text) > 4096:
            self.logger.info(f"Text ({len(text)} chars) exceeds OpenAI limit (4096). Splitting...")
            # Use the base class splitter, but configure it for OpenAI's limit
            self.config.chunk_size = 4000 # Use a slightly smaller chunk size for safety
            text_chunks = self._split_text(text)
            self.logger.info(f"Split into {len(text_chunks)} chunks.")

        self.logger.info(f"Generating audio via OpenAI (model: {self.model_specifier}, voice: {self.config.cloud_voice})")
        start_time = time.time()
        
        audio_files_to_merge = []
        
        try:
             with LoggingProgress(logger, f"OpenAI TTS generation", len(text_chunks)) as progress:
                 for i, chunk in enumerate(text_chunks):
                      # Create a unique temp path for this chunk
                      chunk_temp_path = DEFAULT_CACHE_DIR / "audio_temp" / f"openai_chunk_{int(time.time()*1000)}_{i}.{api_format}"
                      chunk_temp_path.parent.mkdir(parents=True, exist_ok=True)
                      
                      response = self.model_handle.audio.speech.create(
                          model=self.model_specifier,
                          voice=self.config.cloud_voice,
                          input=chunk,
                          response_format=api_format,
                          speed=self.config.speed # API supports speed directly
                      )
                      
                      # Stream response directly to the temporary file
                      response.stream_to_file(str(chunk_temp_path))
                      audio_files_to_merge.append(chunk_temp_path)
                      progress.update(1)

             processing_time = time.time() - start_time
             self.logger.info(f"OpenAI API calls successful (took {processing_time:.2f}s for {len(text_chunks)} chunks)")

             # --- Combine and Post-process ---
             sf = self.post_processor.get_sf()
             
             # 1. Load all chunks
             audio_arrays = []
             api_sample_rate = -1
             for f_path in audio_files_to_merge:
                  try:
                       audio, sr = self.post_processor._load_audio(f_path, sr=None) # Load native SR
                       if api_sample_rate == -1: api_sample_rate = sr
                       if sr != api_sample_rate: # Should not happen with OpenAI, but good to check
                            self.logger.warning(f"Sample rate mismatch in chunks! Expected {api_sample_rate}, got {sr}. Resampling.")
                            librosa = self.post_processor.get_librosa()
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=api_sample_rate)
                       audio_arrays.append(audio)
                  except Exception as load_e:
                       self.logger.error(f"Failed to load downloaded chunk {f_path}: {load_e}")
                  finally:
                       try: f_path.unlink() # Clean up temp chunk file
                       except OSError: pass

             if not audio_arrays:
                  raise GenerationError("Failed to load any audio chunks from OpenAI.", self.model_specifier)
             
             # 2. Combine chunks (using base class helper)
             combined_audio = self._combine_audio(audio_arrays, api_sample_rate)
             if combined_audio is None:
                  raise GenerationError("Failed to combine audio chunks.", self.model_specifier)

             # 3. Post-process (pitch, normalize) and Resample if needed
             # Note: Speed was already applied by the API.
             # We must resample if config.sample_rate != api_sample_rate
             target_sr = self.config.sample_rate
             if api_sample_rate != target_sr:
                 self.logger.info(f"Resampling audio from OpenAI SR {api_sample_rate}Hz to target SR {target_sr}Hz.")
                 librosa = self.post_processor.get_librosa()
                 combined_audio = librosa.resample(combined_audio, orig_sr=api_sample_rate, target_sr=target_sr)
                 
             processed_audio = self.post_processor._post_process_audio(
                 combined_audio,
                 target_sr,
                 speed=1.0, # Speed already applied by API
                 pitch_shift=self.config.pitch_shift,
                 volume_normalize=self.config.volume_normalize
             )
             
             # 4. Save final file in the requested format
             saved_path = self.post_processor._save_audio(
                 final_output_path, 
                 processed_audio, 
                 target_sr, 
                 self.config.output_format
             )
             if not saved_path:
                 return None # Saving failed
                 
             # Get final audio info
             duration = len(processed_audio) / target_sr
             
             result = AudioResult(
                 audio_path=saved_path,
                 duration_seconds=duration,
                 sample_rate=target_sr,
                 num_samples=len(processed_audio),
                 model_used=self.model_identifier,
                 processing_time=processing_time,
                 chunks_processed=len(text_chunks)
             )

             self.logger.info(f"Audio generated: {result.duration_formatted} duration, model: {result.model_used}")
             return result

        except Exception as e:
            self.logger.error(f"OpenAI TTS API call failed: {e}", exc_info=True)
            # Clean up temp chunk files
            for f_path in audio_files_to_merge:
                 if f_path.exists():
                      try: f_path.unlink()
                      except OSError: pass
            raise GenerationError(f"OpenAI TTS API call failed: {e}", self.model_specifier) from e
