"""
LLM Handler Module - Enhanced Version
Manages model loading, optimization, and response generation with advanced quantization and layer splitting
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import gc
import os
from pathlib import Path

from loggerConf import get_logger_conf, log_execution_time, log_resource_usage
from config import ModelConfig, MemoryConfig
from config_base import (
    FALLBACK_MODEL,
    DEFAULT_MODEL,
    CACHE_DIR,
    OFFLOAD_DIR,
    QUANTIZATION_OPTIONS,
    DEFAULT_QUANTIZATION,
    ENABLE_LAYER_SPLITTING,
    DEFAULT_GPU_LAYERS
)
from hyperparameters import HyperparameterConfig
from response_filter import ResponseFilter, DynamicTokenLimitCalculator

logger = get_logger_conf(__name__)

def _get_default_hyperparms():
    # Lazy load default hyperparameters
    from config import get_default_hyperparams
    return get_default_hyperparams()

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    method: str = "8bit"  # none, 4bit, 8bit, 16bit
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = True  # For 4bit
    quant_type: str = "nf4"  # nf4 or fp4 for 4bit
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    
    def to_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Convert to BitsAndBytes configuration"""
        if self.method == "none" or self.method == "16bit":
            return None
        
        if self.method == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=self.use_double_quant,
                bnb_4bit_quant_type=self.quant_type
            )
        elif self.method == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.llm_int8_threshold,
                llm_int8_skip_modules=self.llm_int8_skip_modules,
                llm_int8_enable_fp32_cpu_offload=self.llm_int8_enable_fp32_cpu_offload,
                llm_int8_has_fp16_weight=self.llm_int8_has_fp16_weight
            )
        
        return None


@dataclass
class LayerSplitConfig:
    """Configuration for splitting model layers between devices"""
    enabled: bool = True
    gpu_layers: int = -1  # -1 means auto-detect optimal split
    max_gpu_memory: Dict[int, str] = None  # e.g., {0: "10GB", 1: "10GB"}
    max_cpu_memory: str = "30GB"
    offload_folder: Optional[Path] = None
    offload_state_dict: bool = False
    
    def __post_init__(self):
        if self.max_gpu_memory is None:
            # Auto-detect GPUs and allocate memory
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                self.max_gpu_memory = {i: "10GB" for i in range(num_gpus)}
            else:
                self.max_gpu_memory = {}


@dataclass
class GenerationResult:
    """Result from text generation"""
    raw_output: str
    filtered_output: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    memory_used: float
    device_map: Optional[Dict] = None


class AdvancedModelManager:
    """Enhanced model manager with advanced quantization and layer splitting"""
    def __init__(self, 
                 model_name: str = DEFAULT_MODEL,
                 model_id: Optional[str] = None,
                 quantization_config: Optional[QuantizationConfig] = None,
                 layer_split_config: Optional[LayerSplitConfig] = None,
                 hyperparameters: Optional[HyperparameterConfig] = None,
                 memory_config: Optional[MemoryConfig] = None):
        """Initialize advanced model manager"""
        self.model_name = model_name
        self.model_id = model_id
        
        # Get model config from registry
        if model_id is None:
            model_entry = get_model_config(model_name)
            if model_entry:
                self.model_config = self._entry_to_config(model_entry)
                self.model_id = model_entry.model_id
            else:
                # Fallback
                fallback_entry = get_model_config(FALLBACK_MODEL)
                self.model_config = self._entry_to_config(fallback_entry)
                self.model_id = fallback_entry.model_id
        else:
            # Try to find in registry
            model_entry = get_model_config(model_id)
            if model_entry:
                self.model_config = self._entry_to_config(model_entry)
            else:
                # Create basic config for unknown model
                from config import ModelConfig
                self.model_config = ModelConfig(
                    name=model_id.split('/')[-1],
                    model_id=model_id,
                    supports_thinking=False,
                    max_context=8192,
                    optimal_chunk_size=1000
                )
       
        # Configurations
        self.quant_config = quantization_config or QuantizationConfig(method=DEFAULT_QUANTIZATION)
        self.split_config = layer_split_config or LayerSplitConfig(
            enabled=ENABLE_LAYER_SPLITTING,
            gpu_layers=DEFAULT_GPU_LAYERS
        )
        self.hyperparams = hyperparameters or _get_default_hyperparams()
        self.memory_config = memory_config or MemoryConfig()
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.device = None
        self.accelerator = None
        self.device_map = None
        
        # Helper components
        self.response_filter = ResponseFilter(self.model_config)
        self.token_calculator = DynamicTokenLimitCalculator(self.model_config)
        
        logger.info(f"Initialized AdvancedModelManager for {self.model_id}")
        logger.info(f"  Quantization: {self.quant_config.method}")
        logger.info(f"  Layer splitting: {self.split_config.enabled}")
    
    def _entry_to_config(self, entry):
        """Convert ModelEntry to ModelConfig"""
        from config import ModelConfig
        
        return ModelConfig(
            name=entry.name,
            model_id=entry.model_id,
            supports_thinking=entry.supports_thinking,
            thinking_tokens=entry.thinking_tokens or [],
            max_context=entry.max_context,
            optimal_chunk_size=entry.optimal_chunk_size,
            temperature=entry.temperature,
            top_p=entry.top_p,
            max_new_tokens=entry.max_new_tokens,
            quantization_support=entry.quantization_support or ["4bit", "8bit"]
        )

    @log_execution_time()
    @log_resource_usage()
    def load_model(self, trust_remote_code: bool = False) -> Tuple[Any, Any]:
        """
        Load model with advanced optimizations
        
        Args:
            trust_remote_code: Whether to trust remote code in model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_id}")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            logger.warning("No GPU detected, loading on CPU (will be slower)")
            return self._load_cpu_model(trust_remote_code)
        
        # Load with quantization and layer splitting
        if self.quant_config.method != "none":
            return self._load_quantized_model(trust_remote_code)
        elif self.split_config.enabled:
            return self._load_split_model(trust_remote_code)
        else:
            return self._load_standard_model(trust_remote_code)
    
    def _load_quantized_model(self, trust_remote_code: bool) -> Tuple[Any, Any]:
        """Load model with quantization"""
        logger.info(f"Loading with {self.quant_config.method} quantization")
        
        # Get BitsAndBytes config
        bnb_config = self.quant_config.to_bnb_config()
        
        # Prepare device map
        if self.split_config.enabled and self.split_config.gpu_layers != 0:
            device_map = self._create_device_map()
        else:
            device_map = "auto"
        
        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                max_memory=self.split_config.max_gpu_memory if self.split_config.enabled else None,
                torch_dtype=self.quant_config.compute_dtype,
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR,
                offload_folder=self.split_config.offload_folder or OFFLOAD_DIR,
                offload_state_dict=self.split_config.offload_state_dict,
                trust_remote_code=trust_remote_code
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=CACHE_DIR,
                use_fast=True,
                trust_remote_code=trust_remote_code
            )
            
            # Store device map
            if hasattr(model, 'hf_device_map'):
                self.device_map = model.hf_device_map
            
            self.model = model
            self.tokenizer = tokenizer
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Update hyperparameters with tokenizer info
            if self.hyperparams.pad_token_id is None:
                self.hyperparams.pad_token_id = self.tokenizer.pad_token_id
            if self.hyperparams.eos_token_id is None:
                self.hyperparams.eos_token_id = self.tokenizer.eos_token_id
            if self.hyperparams.bos_token_id is None:
                self.hyperparams.bos_token_id = self.tokenizer.bos_token_id
            
            logger.info(f"Successfully loaded quantized model")
            self._log_model_info()
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            logger.info("Falling back to standard loading")
            return self._load_standard_model(trust_remote_code)
    
    def _load_split_model(self, trust_remote_code: bool) -> Tuple[Any, Any]:
        """Load model with layer splitting (no quantization)"""
        logger.info("Loading model with layer splitting")
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Create device map
        device_map = self._create_device_map()
        
        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=device_map,
                max_memory=self._get_max_memory_dict(),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR,
                offload_folder=self.split_config.offload_folder or OFFLOAD_DIR,
                offload_state_dict=self.split_config.offload_state_dict,
                trust_remote_code=trust_remote_code
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=CACHE_DIR,
                use_fast=True,
                trust_remote_code=trust_remote_code
            )
            
            # Store device map
            if hasattr(model, 'hf_device_map'):
                self.device_map = model.hf_device_map
            
            self.model = model
            self.tokenizer = tokenizer
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Update hyperparameters
            if self.hyperparams.pad_token_id is None:
                self.hyperparams.pad_token_id = self.tokenizer.pad_token_id
            if self.hyperparams.eos_token_id is None:
                self.hyperparams.eos_token_id = self.tokenizer.eos_token_id
            if self.hyperparams.bos_token_id is None:
                self.hyperparams.bos_token_id = self.tokenizer.bos_token_id
            
            logger.info("Successfully loaded model with layer splitting")
            self._log_model_info()
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load split model: {e}")
            return self._load_standard_model(trust_remote_code)
    
    def _load_standard_model(self, trust_remote_code: bool) -> Tuple[Any, Any]:
        """Load model without special optimizations"""
        logger.info("Loading model in standard mode")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=CACHE_DIR,
                use_fast=True,
                trust_remote_code=trust_remote_code
            )
            
            self.model = model
            self.tokenizer = tokenizer
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Update hyperparameters
            if self.hyperparams.pad_token_id is None:
                self.hyperparams.pad_token_id = self.tokenizer.pad_token_id
            if self.hyperparams.eos_token_id is None:
                self.hyperparams.eos_token_id = self.tokenizer.eos_token_id
            if self.hyperparams.bos_token_id is None:
                self.hyperparams.bos_token_id = self.tokenizer.bos_token_id
            
            logger.info("Successfully loaded model in standard mode")
            self._log_model_info()
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_cpu_model(self, trust_remote_code: bool) -> Tuple[Any, Any]:
        """Load model on CPU only"""
        logger.info("Loading model on CPU")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=CACHE_DIR,
                use_fast=True,
                trust_remote_code=trust_remote_code
            )
            
            self.model = model
            self.tokenizer = tokenizer
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Update hyperparameters
            if self.hyperparams.pad_token_id is None:
                self.hyperparams.pad_token_id = self.tokenizer.pad_token_id
            if self.hyperparams.eos_token_id is None:
                self.hyperparams.eos_token_id = self.tokenizer.eos_token_id
            if self.hyperparams.bos_token_id is None:
                self.hyperparams.bos_token_id = self.tokenizer.bos_token_id
            
            logger.info("Successfully loaded model on CPU")
            self._log_model_info()
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load CPU model: {e}")
            raise
    
    def _create_device_map(self) -> Dict[str, Any]:
        """Create optimal device map for layer splitting"""
        if not self.split_config.enabled:
            return "auto"
        
        if self.split_config.gpu_layers == -1:
            # Auto-detect optimal split
            return "auto"
        elif self.split_config.gpu_layers == 0:
            # CPU only
            return {"": "cpu"}
        else:
            # Manual split - need to inspect model architecture
            # This is a simplified version; actual implementation would need
            # to analyze the model's layer structure
            try:
                with init_empty_weights():
                    temp_model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        cache_dir=CACHE_DIR
                    )
                
                device_map = infer_auto_device_map(
                    temp_model,
                    max_memory=self._get_max_memory_dict(),
                    dtype=torch.bfloat16
                )
                
                logger.debug(f"Inferred device map: {device_map}")
                return device_map
                
            except Exception as e:
                logger.warning(f"Could not infer device map: {e}, using auto")
                return "auto"
    
    def _get_max_memory_dict(self) -> Dict:
        """Get max memory dictionary for device mapping"""
        max_memory = {}
        
        # GPU memory
        if self.split_config.max_gpu_memory:
            max_memory.update(self.split_config.max_gpu_memory)
        
        # CPU memory
        max_memory["cpu"] = self.split_config.max_cpu_memory
        
        return max_memory
    
    def _log_model_info(self):
        """Log information about loaded model"""
        if self.model is None:
            return
        
        # Get model size
        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = (param_count * 2) / (1024 * 1024)  # Assuming fp16
        
        logger.info(f"Model loaded: {param_count:,} parameters ({param_size_mb:.1f} MB)")
        
        # Log device map if available
        if self.device_map:
            logger.info("Device map:")
            device_counts = {}
            for layer, device in self.device_map.items():
                device_str = str(device)
                device_counts[device_str] = device_counts.get(device_str, 0) + 1
            
            for device, count in device_counts.items():
                logger.info(f"  {device}: {count} layers")
        
        # Log memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                logger.info(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    @log_execution_time()
    def generate(self,
                prompt: str,
                hyperparams: Optional[HyperparameterConfig] = None,
                remove_thinking: bool = True) -> GenerationResult:
        """
        Generate text from prompt using configured hyperparameters
        
        Args:
            prompt: Input prompt
            hyperparams: Override hyperparameters (uses instance config if None)
            remove_thinking: Whether to filter thinking tokens
            
        Returns:
            GenerationResult with generated text and metadata
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use provided hyperparams or fall back to instance config
        hp = hyperparams or self.hyperparams
        
        # Validate hyperparameters
        errors = hp.validate()
        if errors:
            logger.warning(f"Hyperparameter validation errors: {errors}")
        
        # Calculate optimal max_new_tokens if not specified
        if hp.max_new_tokens is None:
            hp.max_new_tokens = self.token_calculator.calculate_max_new_tokens(prompt)
        
        logger.debug(f"Generating with hyperparameters: {hp.to_dict()}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_tokens = inputs['input_ids'].shape[1]
        
        # Move to correct device
        device = self._get_model_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create generation config from hyperparameters
        gen_config = GenerationConfig(**hp.to_dict())
        
        # Generate
        import time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode output
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated portion
        raw_output = full_output[len(prompt):].strip() if full_output.startswith(prompt) else full_output
        
        # Filter if requested
        if remove_thinking and self.model_config.supports_thinking:
            filter_result = self.response_filter.filter(raw_output)
            filtered_output = filter_result.filtered_text
            logger.debug(f"Filtered {filter_result.removal_ratio:.1%} of output")
        else:
            filtered_output = raw_output
        
        # Get memory usage
        memory_used = 0
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        output_tokens = outputs.shape[1] - input_tokens
        
        logger.info(f"Generated {output_tokens} tokens in {generation_time:.2f}s "
                   f"({output_tokens/generation_time:.1f} tokens/s)")
        
        return GenerationResult(
            raw_output=raw_output,
            filtered_output=filtered_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_time=generation_time,
            memory_used=memory_used,
            device_map=self.device_map
        )
    
    def _get_model_device(self) -> torch.device:
        """Get the device of the first model parameter"""
        if hasattr(self.model, 'hf_device_map'):
            # Model has device map, get first layer's device
            return next(self.model.parameters()).device
        return self.device or torch.device("cpu")
    
    def process_with_chat_template(self,
                                  system_prompt: str,
                                  user_message: str,
                                  hyperparams: Optional[HyperparameterConfig] = None,
                                  **kwargs) -> GenerationResult:
        """
        Process text using chat template
        
        Args:
            system_prompt: System instruction
            user_message: User input
            hyperparams: Override hyperparameters
            **kwargs: Additional arguments
            
        Returns:
            GenerationResult
        """
        # Format as chat
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Try to use chat template
        try:
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback formatting
                prompt = f"{system_prompt}\n\n{user_message}\n\nResponse:"
        
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, using fallback")
            prompt = f"{system_prompt}\n\n{user_message}\n\nResponse:"
        
        return self.generate(prompt, hyperparams=hyperparams, **kwargs)
    
    def update_hyperparameters(self, **kwargs):
        """Update hyperparameters dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.hyperparams, key):
                setattr(self.hyperparams, key, value)
                logger.debug(f"Updated hyperparameter: {key}={value}")
            else:
                logger.warning(f"Unknown hyperparameter: {key}")
    
    def unload_model(self):
        """Unload model and free memory"""
        logger.info("Unloading model and freeing memory")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        logger.info("Model unloaded and memory freed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.unload_model()


# Backward compatibility - use AdvancedModelManager as ModelManager
ModelManager = AdvancedModelManager


class BatchProcessor:
    """Process text in batches for efficiency"""
    
    def __init__(self, model_manager: AdvancedModelManager):
        self.model_manager = model_manager
        self.logger = get_logger_conf(f"{__name__}.BatchProcessor")
    
    def process_batch(self,
                      texts: List[str],
                      system_prompt: str,
                      hyperparams: Optional[HyperparameterConfig] = None,
                      **generation_kwargs) -> List[GenerationResult]:
        """
        Process multiple texts in batches
        
        Args:
            texts: List of texts to process
            system_prompt: System prompt for all texts
            hyperparams: Hyperparameter configuration
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of GenerationResults
        """
        batch_size = 1  # Process one at a time for now
        results = []
        
        self.logger.info(f"Processing {len(texts)} texts")
        
        for i, text in enumerate(texts):
            self.logger.debug(f"Processing text {i+1}/{len(texts)}")
            
            try:
                result = self.model_manager.process_with_chat_template(
                    system_prompt,
                    text,
                    hyperparams=hyperparams,
                    **generation_kwargs
                )
                results.append(result)
            
            except Exception as e:
                self.logger.error(f"Failed to process text {i}: {e}")
                # Create error result
                results.append(GenerationResult(
                    raw_output="",
                    filtered_output=f"Error: {str(e)}",
                    input_tokens=0,
                    output_tokens=0,
                    generation_time=0,
                    memory_used=0
                ))
            
            # Clear cache between items
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.logger.info(f"Batch processing complete: {len(results)} results")
        return results
