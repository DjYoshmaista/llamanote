"""
Audio Generator Module
Text-to-speech generation using HuggingFace models
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import soundfile as sf
import librosa
from transformers import pipeline, AutoProcessor, AutoModel
import warnings

from loggerConf import get_logger_conf, log_execution_time, ConsoleOutput
from huggingface_search import ModelDownloader, ModelInfo
from config_manager import ConfigManager

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger_conf(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio generation"""
    model_id: str = "microsoft/speecht5_tts"
    speaker_embedding: Optional[str] = None
    sample_rate: int = 16000
    output_format: str = "wav"
    chunk_size: int = 5000  # Characters per audio chunk
    speed: float = 1.0
    pitch_shift: int = 0  # Semitones
    volume_normalize: bool = True
    device: str = "auto"  # auto, cpu, cuda
    use_half_precision: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "speaker_embedding": self.speaker_embedding,
            "sample_rate": self.sample_rate,
            "output_format": self.output_format,
            "chunk_size": self.chunk_size,
            "speed": self.speed,
            "pitch_shift": self.pitch_shift,
            "volume_normalize": self.volume_normalize,
            "device": self.device,
            "use_half_precision": self.use_half_precision
        }


@dataclass
class AudioResult:
    """Result from audio generation"""
    audio_path: Path
    duration_seconds: float
    sample_rate: int
    num_samples: int
    model_used: str
    processing_time: float
    chunks_processed: int
    
    @property
    def duration_formatted(self) -> str:
        """Get formatted duration string"""
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes}:{seconds:02d}"


class AudioGenerator:
    """Generate audio from text using various TTS models"""
    
    SUPPORTED_MODELS = {
        "speecht5": ["microsoft/speecht5_tts"],
        "bark": ["suno/bark", "suno/bark-small"],
        "mms": ["facebook/mms-tts-eng"],
        "tacotron2": ["speechbrain/tts-tacotron2-ljspeech"],
        "fastspeech2": ["facebook/fastspeech2-en-ljspeech"],
    }
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio generator
        
        Args:
            config: Audio generation configuration
        """
        self.config = config or AudioConfig()
        self.model = None
        self.processor = None
        self.vocoder = None
        self.pipeline = None
        self.device = self._setup_device()
        self.model_downloader = ModelDownloader()
        self.config_manager = ConfigManager()
        
        logger.info(f"Initialized AudioGenerator with model: {self.config.model_id}")
        
    def _setup_device(self) -> str:
        """Setup compute device"""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
        
    @log_execution_time()
    def load_model(self, model_id: Optional[str] = None) -> bool:
        """
        Load TTS model
        
        Args:
            model_id: Model identifier (uses config if not provided)
            
        Returns:
            Success status
        """
        model_id = model_id or self.config.model_id
        
        logger.info(f"Loading TTS model: {model_id}")
        
        try:
            # Download model if needed
            model_path = self.model_downloader.download_model(model_id)
            if not model_path:
                logger.error(f"Failed to download model: {model_id}")
                return False
                
            # Detect model type and load accordingly
            if "bark" in model_id.lower():
                self._load_bark_model(model_id)
            elif "speecht5" in model_id.lower():
                self._load_speecht5_model(model_id)
            elif "mms" in model_id.lower():
                self._load_mms_model(model_id)
            elif "tacotron" in model_id.lower():
                self._load_tacotron_model(model_id)
            else:
                # Try generic pipeline
                self._load_generic_model(model_id)
                
            logger.info(f"Model loaded successfully: {model_id}")
            self.config.model_id = model_id
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
            
    def _load_bark_model(self, model_id: str):
        """Load Bark model"""
        from transformers import BarkModel, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = BarkModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.config.use_half_precision else torch.float32
        ).to(self.device)
        
    def _load_speecht5_model(self, model_id: str):
        """Load SpeechT5 model"""
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset
        
        self.processor = SpeechT5Processor.from_pretrained(model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_id).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0).to(self.device)
        
    def _load_mms_model(self, model_id: str):
        """Load MMS model"""
        from transformers import VitsModel, AutoTokenizer
        
        self.model = VitsModel.from_pretrained(model_id).to(self.device)
        self.processor = AutoTokenizer.from_pretrained(model_id)
        
    def _load_tacotron_model(self, model_id: str):
        """Load Tacotron2 model"""
        # Use pipeline for simplicity
        self.pipeline = pipeline(
            "text-to-speech",
            model=model_id,
            device=0 if self.device == "cuda" else -1
        )
        
    def _load_generic_model(self, model_id: str):
        """Load generic TTS model using pipeline"""
        self.pipeline = pipeline(
            "text-to-speech",
            model=model_id,
            device=0 if self.device == "cuda" else -1
        )
        
    @log_execution_time()
    def generate_audio(self,
                      text: str,
                      output_path: Optional[Path] = None,
                      chunk_text: bool = True) -> Optional[AudioResult]:
        """
        Generate audio from text
        
        Args:
            text: Input text
            output_path: Output file path
            chunk_text: Whether to process in chunks
            
        Returns:
            AudioResult or None if failed
        """
        if not self.model and not self.pipeline:
            logger.error("No model loaded")
            return None
            
        logger.info(f"Generating audio for {len(text)} characters")
        
        try:
            import time
            start_time = time.time()
            
            # Process text in chunks if needed
            if chunk_text and len(text) > self.config.chunk_size:
                audio_arrays = self._generate_chunked(text)
            else:
                audio_arrays = [self._generate_single(text)]
                
            # Combine audio chunks
            audio = self._combine_audio(audio_arrays)
            
            # Post-process audio
            audio = self._post_process_audio(audio)
            
            # Save audio
            if output_path is None:
                output_path = Path("output.wav")
                
            output_path = self._save_audio(audio, output_path)
            
            # Calculate statistics
            duration = len(audio) / self.config.sample_rate
            processing_time = time.time() - start_time
            
            result = AudioResult(
                audio_path=output_path,
                duration_seconds=duration,
                sample_rate=self.config.sample_rate,
                num_samples=len(audio),
                model_used=self.config.model_id,
                processing_time=processing_time,
                chunks_processed=len(audio_arrays)
            )
            
            logger.info(f"Audio generated: {result.duration_formatted} duration, "
                       f"{result.chunks_processed} chunks")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            return None
            
    def _generate_single(self, text: str) -> np.ndarray:
        """Generate audio for a single text chunk"""
        if "bark" in self.config.model_id.lower():
            return self._generate_bark(text)
        elif "speecht5" in self.config.model_id.lower():
            return self._generate_speecht5(text)
        elif "mms" in self.config.model_id.lower():
            return self._generate_mms(text)
        elif self.pipeline:
            return self._generate_pipeline(text)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_id}")
            
    def _generate_bark(self, text: str) -> np.ndarray:
        """Generate using Bark model"""
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            audio_values = self.model.generate(**inputs)
            
        return audio_values.cpu().numpy().squeeze()
        
    def _generate_speecht5(self, text: str) -> np.ndarray:
        """Generate using SpeechT5 model"""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embeddings,
                vocoder=self.vocoder
            )
            
        return speech.cpu().numpy()
        
    def _generate_mms(self, text: str) -> np.ndarray:
        """Generate using MMS model"""
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model(**inputs).waveform
            
        return output.cpu().numpy().squeeze()
        
    def _generate_pipeline(self, text: str) -> np.ndarray:
        """Generate using pipeline"""
        output = self.pipeline(text)
        
        if isinstance(output, dict) and "audio" in output:
            return output["audio"]
        elif isinstance(output, np.ndarray):
            return output
        else:
            return np.array(output)
            
    def _generate_chunked(self, text: str) -> List[np.ndarray]:
        """Generate audio in chunks"""
        chunks = self._split_text(text)
        audio_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            audio = self._generate_single(chunk)
            audio_chunks.append(audio)
            
            ConsoleOutput.progress_bar(i+1, len(chunks), prefix="Generating audio")
            
        return audio_chunks
        
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        chunks = []
        
        # Split by sentences first
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.config.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
        
    def _combine_audio(self, audio_arrays: List[np.ndarray]) -> np.ndarray:
        """Combine multiple audio arrays"""
        if len(audio_arrays) == 1:
            return audio_arrays[0]
            
        # Add small silence between chunks
        silence_samples = int(0.2 * self.config.sample_rate)  # 200ms
        silence = np.zeros(silence_samples)
        
        combined = []
        for i, audio in enumerate(audio_arrays):
            combined.append(audio)
            if i < len(audio_arrays) - 1:
                combined.append(silence)
                
        return np.concatenate(combined)
        
    def _post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Post-process audio (speed, pitch, normalization)"""
        # Speed adjustment
        if self.config.speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=self.config.speed)
            
        # Pitch shift
        if self.config.pitch_shift != 0:
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.config.sample_rate,
                n_steps=self.config.pitch_shift
            )
            
        # Volume normalization
        if self.config.volume_normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val * 0.95  # Normalize to 95% to avoid clipping
                
        return audio
        
    def _save_audio(self, audio: np.ndarray, output_path: Path) -> Path:
        """Save audio to file"""
        output_path = Path(output_path)
        
        # Ensure extension matches format
        if self.config.output_format == "mp3":
            output_path = output_path.with_suffix(".mp3")
        elif self.config.output_format == "flac":
            output_path = output_path.with_suffix(".flac")
        else:
            output_path = output_path.with_suffix(".wav")
            
        # Save audio
        sf.write(
            output_path,
            audio,
            self.config.sample_rate,
            subtype='PCM_16' if self.config.output_format == "wav" else None
        )
        
        logger.info(f"Audio saved to {output_path}")
        return output_path
        
    def list_available_models(self) -> List[str]:
        """List available TTS models"""
        models = []
        for category, model_list in self.SUPPORTED_MODELS.items():
            models.extend(model_list)
        return models
        
    def estimate_duration(self, text: str) -> float:
        """Estimate audio duration for text"""
        # Rough estimation: 150 words per minute
        word_count = len(text.split())
        minutes = word_count / 150
        return minutes * 60  # Return seconds
        
    def unload_model(self):
        """Unload current model to free memory"""
        if self.model:
            del self.model
            self.model = None
            
        if self.processor:
            del self.processor
            self.processor = None
            
        if self.vocoder:
            del self.vocoder
            self.vocoder = None
            
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Model unloaded")


class AudioPostProcessor:
    """Post-process audio files"""
    
    @staticmethod
    def add_background_music(speech_path: Path, 
                            music_path: Path,
                            output_path: Path,
                            music_volume: float = 0.2) -> Path:
        """Add background music to speech"""
        speech, sr1 = librosa.load(speech_path, sr=None)
        music, sr2 = librosa.load(music_path, sr=None)
        
        # Resample if needed
        if sr1 != sr2:
            music = librosa.resample(music, orig_sr=sr2, target_sr=sr1)
            
        # Loop music if shorter than speech
        if len(music) < len(speech):
            repeats = (len(speech) // len(music)) + 1
            music = np.tile(music, repeats)
            
        # Trim music to match speech length
        music = music[:len(speech)]
        
        # Mix
        mixed = speech + (music * music_volume)
        
        # Normalize
        mixed = mixed / np.abs(mixed).max() * 0.95
        
        # Save
        sf.write(output_path, mixed, sr1)
        return output_path
        
    @staticmethod
    def add_silence(audio_path: Path, 
                   start_silence: float = 0.5,
                   end_silence: float = 1.0) -> Path:
        """Add silence to beginning and end"""
        audio, sr = librosa.load(audio_path, sr=None)
        
        start_samples = int(start_silence * sr)
        end_samples = int(end_silence * sr)
        
        audio = np.concatenate([
            np.zeros(start_samples),
            audio,
            np.zeros(end_samples)
        ])
        
        sf.write(audio_path, audio, sr)
        return audio_path
        
    @staticmethod
    def apply_effects(audio_path: Path,
                     reverb: float = 0.0,
                     echo: float = 0.0) -> Path:
        """Apply audio effects"""
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Simple reverb (convolution with exponential decay)
        if reverb > 0:
            ir_length = int(reverb * sr)
            ir = np.exp(-3 * np.linspace(0, 1, ir_length))
            ir = ir * np.random.randn(ir_length) * 0.1
            audio = np.convolve(audio, ir, mode='same')
            
        # Simple echo
        if echo > 0:
            delay_samples = int(echo * sr)
            echo_audio = np.zeros_like(audio)
            echo_audio[delay_samples:] = audio[:-delay_samples] * 0.5
            audio = audio + echo_audio
            
        # Normalize
        audio = audio / np.abs(audio).max() * 0.95
        
        sf.write(audio_path, audio, sr)
        return audio_path
