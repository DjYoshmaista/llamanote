# llamanote/models/backends/openai_audio.py
"""
OpenAI Text-to-Speech (TTS) Backend
"""

import time
from pathlib import Path
from typing import Optional, Dict

from .base import AudioBackend
from ...core.types import AudioConfig, AudioResult
from ...utils.logger import get_logger_conf, ConsoleOutput
from ...utils.decorators import log_execution_time
from ...processing.audio_processor import AudioPostProcessor # Import for post-processing
from ...core.errors import ModelLoadError, GenerationError

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

class OpenAITTSBackend(AudioBackend):
    """Generate audio using OpenAI's Text-to-Speech API."""

    def __init__(self,
                 config: AudioConfig,
                 model_specifier: str,
                 api_key: str):
        
        super().__init__(provider_id="openai_audio",
                         model_specifier=model_specifier,
                         config=config)
                         
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run 'pip install openai'")
            
        self.api_key = api_key
        # self.model_handle (defined in base class) will hold the OpenAI client
        
        # Validate model specifier
        if model_specifier not in ["tts-1", "tts-1-hd", "tts-1-1106", "tts-1-hd-1106"]: # Add newer ones
             logger.warning(f"Unsupported OpenAI TTS model '{model_specifier}', defaulting to 'tts-1'.")
             self.model_specifier = "tts-1"
             
        self.voice = self.config.cloud_voice or "alloy" # Default to 'alloy' if not set

    def load(self, **kwargs) -> bool:
        """Initialize the OpenAI client and test the API key."""
        if self.model_handle:
            return True
        try:
            self.model_handle = OpenAI(api_key=self.api_key)
            self.model_handle.models.list() # Test API key
            logger.info(f"OpenAI (TTS) API client initialized for model {self.model_specifier}")
            return True
        except AuthenticationError as e:
             logger.error(f"OpenAI API key is invalid: {e}")
             ConsoleOutput.error("OpenAI API key is invalid. Please check your configuration.")
             raise ModelLoadError(f"OpenAI API key invalid: {e}", self.model_specifier) from e
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client or verify key: {e}", exc_info=True)
            ConsoleOutput.error(f"OpenAI connection failed: {e}. Check key and network access.")
            raise ModelLoadError(f"OpenAI connection failed: {e}", self.model_specifier) from e

    def unload(self):
        """No explicit unloading needed, just clear client reference."""
        if self.model_handle is not None:
             del self.model_handle
             self.model_handle = None
             logger.debug("OpenAI (TTS) client reference cleared.")

    @log_execution_time(logger_name=__name__)
    def generate_audio(self,
                      text: str,
                      output_path: Path, # Now expects a full Path object
                      **kwargs) -> Optional[AudioResult]:
        """
        Generate audio using OpenAI API.
        
        Args:
            text: Text to synthesize.
            output_path: The *final* desired output path (e.g., .../file.wav)
            **kwargs: Ignored (like chunk_text)
            
        Returns:
            AudioResult or None on failure.
        """
        if self.model_handle is None:
            logger.error("OpenAI client not initialized. Call load() first.")
            return None
        
        # API supports specific formats. 'wav' is not one. Let's use 'mp3' as intermediate.
        api_format = "mp3" 
        temp_output_path = output_path.with_suffix(f".{api_format}")
        
        # Check if file already exists
        if output_path.exists():
            logger.warning(f"Output file {output_path} already exists. Overwriting.")
        if temp_output_path.exists() and temp_output_path != output_path:
             temp_output_path.unlink() # Clean up previous temp file

        logger.info(f"Generating audio via OpenAI TTS (model: {self.model_specifier}, voice: {self.voice})")
        start_time = time.time()

        try:
            # API call - streams response directly to file
            response = self.model_handle.audio.speech.create(
                model=self.model_specifier,
                voice=self.voice,
                input=text,
                response_format=api_format,
                speed=self.config.speed # API supports speed directly
            )
            
            response.stream_to_file(str(temp_output_path))
            api_time = time.time() - start_time
            logger.info(f"OpenAI TTS API call successful (took {api_time:.2f}s)")

            # Post-process (pitch, normalize) and convert format if necessary
            final_audio_path = self._handle_postprocessing_and_format(
                temp_output_path, 
                output_path, # The final desired path
                api_format # The format we got from the API
            )
            
            if not final_audio_path or not final_audio_path.exists():
                 raise GenerationError("Post-processing or format conversion failed.", self.model_specifier)

            # Get audio info from the *final* saved file
            sf = get_sf()
            audio_info = sf.info(str(final_audio_path))
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
            num_samples = audio_info.frames

            result = AudioResult(
                audio_path=final_audio_path.resolve(),
                duration_seconds=duration,
                sample_rate=sample_rate,
                num_samples=num_samples,
                model_used=self.model_identifier,
                processing_time=time.time() - start_time, # Total time
                chunks_processed=1 # API handles input length
            )

            logger.info(f"Audio generated: {result.duration_formatted} duration, model: {result.model_used}")
            return result

        except Exception as e:
            logger.error(f"OpenAI TTS API call or processing failed: {e}", exc_info=True)
            # Clean up temp file
            if temp_output_path.exists() and temp_output_path != output_path:
                 try: temp_output_path.unlink() 
                 except OSError: pass
            return None # Indicate failure

    def _handle_postprocessing_and_format(self, 
                                          input_audio_path: Path, 
                                          target_output_path: Path,
                                          input_format: str) -> Optional[Path]:
         """Apply post-processing and ensure final format."""
         target_format = self.config.output_format.lower()
         
         needs_postprocessing = (self.config.pitch_shift != 0 or self.config.volume_normalize)
         needs_format_conversion = (input_format != target_format)

         if not needs_postprocessing and not needs_format_conversion:
              if input_audio_path != target_output_path:
                   return input_audio_path.rename(target_output_path)
              return input_audio_path

         # Load audio for processing/conversion
         try:
              librosa = get_librosa()
              # Load, resampling to config SR *only if* post-processing is needed,
              # as resampling might not be desired if only format conversion is happening.
              # However, pitch shift and normalization require a known SR.
              # Let's load at the config's target SR.
              audio, sr = self._load_audio(input_audio_path, sr=self.config.sample_rate)
              current_sr = sr # This is now self.config.sample_rate

              processed_audio = audio.astype(np.float32) # Ensure float for processing

              # Apply post-processing (only those not handled by API)
              if self.config.pitch_shift != 0:
                   logger.debug(f"Applying pitch shift: {self.config.pitch_shift} semitones")
                   processed_audio = librosa.effects.pitch_shift(
                       y=processed_audio, sr=current_sr, n_steps=self.config.pitch_shift
                   )
              
              if self.config.volume_normalize:
                   logger.debug("Applying volume normalization.")
                   processed_audio = AudioPostProcessor._normalize(processed_audio, 0.95)

              # Save in the target format
              # Ensure correct suffix for saving
              final_output_path = target_output_path.with_suffix(f".{target_format}")
              
              subtype = 'PCM_16' if target_format == "wav" else None # Good default for wav
              
              self._save_audio(final_output_path, processed_audio, current_sr, 
                               format=target_format, subtype=subtype)

              # Clean up temporary input file if different from final path
              if input_audio_path != final_output_path and input_audio_path.exists():
                   try: input_audio_path.unlink()
                   except OSError: pass

              return final_output_path

         except Exception as e:
              logger.error(f"Error during audio post-processing/conversion: {e}", exc_info=True)
              if input_audio_path.exists() and input_audio_path != target_output_path:
                   try: input_audio_path.unlink()
                   except OSError: pass
              return None
