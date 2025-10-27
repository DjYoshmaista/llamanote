"""
LLM Handler Module
Manages model loading, optimization, and response generation
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import gc
import os
from pathlib import Path

from logging_config import get_logger, log_execution_time, log_resource_usage
from config import (
    ModelConfig, 
    MemoryConfig, 
    MODELS, 
    DEFAULT_MODEL,
    CACHE_DIR,
    OFFLOAD_DIR
)
from response_filter import ResponseFilter, DynamicTokenLimitCalculator

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Result from text generation"""
    raw_output: str
    filtered_output: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    memory_used: float


class ModelManager:
    """Manages model loading, optimization, and lifecycle"""
    
    def __init__(self, 
                 model_name: str = DEFAULT_MODEL,
                 memory_config: Optional[MemoryConfig] = None):
        """
        Initialize model manager
        
        Args:
            model_name: Name of model from config
            memory_config: Memory optimization configuration
        """
        self.model_name = model_name
        self.model_config = MODELS.get(model_name)
        
        if not self.model_config:
            logger.warning(f"Unknown model {model_name}, using default")
            self.model_config = MODELS[DEFAULT_MODEL]
            
        self.memory_config = memory_config or MemoryConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.accelerator = None
        self.response_filter = ResponseFilter(self.model_config)
        self.token_calculator = DynamicTokenLimitCalculator(self.model_config)
        
        logger.info(f"Initialized ModelManager for {self.model_config.name}")
        
    @log_execution_time()
    @log_resource_usage()
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load model with optimizations
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.model_config.model_id}")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load based on configuration
        if self.memory_config.use_quantization and self.device == "cuda":
            self.model, self.tokenizer = self._load_quantized_model()
        else:
            self.model, self.tokenizer = self._load_standard_model()
            
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")
            
        # Log model info
        self._log_model_info()
        
        return self.model, self.tokenizer
        
    def _load_quantized_model(self) -> Tuple[Any, Any]:
        """Load model with quantization"""
        logger.info(f"Loading with {self.memory_config.quantization_type} quantization")
        
        # Configure quantization
        if self.memory_config.quantization_type == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.memory_config.quantization_type == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
            
        # Parse memory limits
        max_memory = self._parse_memory_config()
        
        try:
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                cache_dir=CACHE_DIR,
                offload_folder=OFFLOAD_DIR if self.memory_config.offload_to_disk else None,
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_id,
                cache_dir=CACHE_DIR,
                use_fast=True
            )
            
            logger.info(f"Successfully loaded quantized model")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            logger.info("Falling back to standard loading")
            return self._load_standard_model()
            
    def _load_standard_model(self) -> Tuple[Any, Any]:
        """Load model without quantization but with device mapping"""
        logger.info("Loading model with automatic device mapping")
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Parse memory limits
        max_memory = self._parse_memory_config()
        
        try:
            # Try to infer optimal device map
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model_id,
                    torch_dtype=torch.bfloat16,
                    cache_dir=CACHE_DIR
                )
                
            device_map = infer_auto_device_map(
                empty_model,
                max_memory=max_memory,
                dtype=torch.bfloat16
            )
            logger.debug(f"Inferred device map: {device_map}")
            
        except Exception as e:
            logger.warning(f"Could not infer device map: {e}")
            device_map = "auto"
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR,
            offload_folder=OFFLOAD_DIR if self.memory_config.offload_to_disk else None,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_id,
            cache_dir=CACHE_DIR,
            use_fast=True
        )
        
        # Prepare with accelerator if available
        if self.accelerator:
            model, tokenizer = self.accelerator.prepare(model, tokenizer)
            
        logger.info("Successfully loaded model with device mapping")
        return model, tokenizer
        
    def _parse_memory_config(self) -> Dict[int, str]:
        """Parse memory configuration into format for transformers"""
        max_memory = {}
        
        # GPU memory
        if torch.cuda.is_available() and self.memory_config.max_gpu_memory != "0GB":
            for i in range(torch.cuda.device_count()):
                max_memory[i] = self.memory_config.max_gpu_memory
                
        # CPU memory
        max_memory["cpu"] = self.memory_config.max_cpu_memory
        
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
        if hasattr(self.model, 'hf_device_map'):
            device_map = self.model.hf_device_map
            logger.debug(f"Device map: {device_map}")
            
            # Count layers per device
            device_counts = {}
            for layer, device in device_map.items():
                device_str = str(device)
                device_counts[device_str] = device_counts.get(device_str, 0) + 1
                
            for device, count in device_counts.items():
                logger.info(f"  {count} layers on {device}")
                
    @log_execution_time()
    def generate(self,
                prompt: str,
                max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                do_sample: bool = True,
                remove_thinking: bool = True) -> GenerationResult:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            remove_thinking: Whether to filter thinking tokens
            
        Returns:
            GenerationResult with generated text and metadata
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Calculate optimal token limit if not specified
        if max_new_tokens is None:
            max_new_tokens = self.token_calculator.calculate_max_new_tokens(prompt)
        else:
            max_new_tokens = min(max_new_tokens, 
                                self.token_calculator.calculate_max_new_tokens(prompt))
            
        # Use model's default parameters if not specified
        temperature = temperature or self.model_config.temperature
        top_p = top_p or self.model_config.top_p
        
        logger.debug(f"Generating with max_new_tokens={max_new_tokens}, "
                    f"temperature={temperature}, top_p={top_p}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_tokens = inputs['input_ids'].shape[1]
        
        # Move to correct device
        device = self._get_model_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Set up generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
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
            memory_used=memory_used
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
                                  **kwargs) -> GenerationResult:
        """
        Process text using chat template
        
        Args:
            system_prompt: System instruction
            user_message: User input
            **kwargs: Additional generation parameters
            
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
            
        return self.generate(prompt, **kwargs)
        
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
        
    def switch_model(self, model_name: str):
        """Switch to a different model"""
        logger.info(f"Switching from {self.model_name} to {model_name}")
        
        # Unload current model
        self.unload_model()
        
        # Update configuration
        self.model_name = model_name
        self.model_config = MODELS.get(model_name)
        
        if not self.model_config:
            logger.error(f"Unknown model {model_name}")
            raise ValueError(f"Unknown model: {model_name}")
            
        # Update filter and calculator
        self.response_filter = ResponseFilter(self.model_config)
        self.token_calculator = DynamicTokenLimitCalculator(self.model_config)
        
        # Load new model
        self.load_model()
        
    def __del__(self):
        """Cleanup on deletion"""
        self.unload_model()


class BatchProcessor:
    """Process text in batches for efficiency"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = get_logger(f"{__name__}.BatchProcessor")
        
    def process_batch(self,
                      texts: List[str],
                      system_prompt: str,
                      batch_size: Optional[int] = None,
                      **generation_kwargs) -> List[GenerationResult]:
        """
        Process multiple texts in batches
        
        Args:
            texts: List of texts to process
            system_prompt: System prompt for all texts
            batch_size: Batch size (from config if not specified)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of GenerationResults
        """
        batch_size = batch_size or self.model_manager.memory_config.batch_size
        results = []
        
        self.logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_results = []
            for text in batch:
                try:
                    result = self.model_manager.process_with_chat_template(
                        system_prompt,
                        text,
                        **generation_kwargs
                    )
                    batch_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process text: {e}")
                    # Create error result
                    batch_results.append(GenerationResult(
                        raw_output="",
                        filtered_output=f"Error: {str(e)}",
                        input_tokens=0,
                        output_tokens=0,
                        generation_time=0,
                        memory_used=0
                    ))
                    
            results.extend(batch_results)
            
            # Clear cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        self.logger.info(f"Batch processing complete: {len(results)} results")
        return results


class ModelPool:
    """Manage multiple models for different tasks"""
    
    def __init__(self, memory_config: Optional[MemoryConfig] = None):
        self.memory_config = memory_config
        self.models: Dict[str, ModelManager] = {}
        self.active_model: Optional[str] = None
        self.logger = get_logger(f"{__name__}.ModelPool")
        
    def add_model(self, name: str, model_name: str) -> ModelManager:
        """Add a model to the pool"""
        if name in self.models:
            self.logger.warning(f"Model {name} already exists, replacing")
            self.models[name].unload_model()
            
        model_manager = ModelManager(model_name, self.memory_config)
        self.models[name] = model_manager
        
        self.logger.info(f"Added model {name} ({model_name}) to pool")
        return model_manager
        
    def get_model(self, name: str, load_if_needed: bool = True) -> Optional[ModelManager]:
        """Get a model from the pool"""
        if name not in self.models:
            self.logger.error(f"Model {name} not in pool")
            return None
            
        model_manager = self.models[name]
        
        if load_if_needed and model_manager.model is None:
            self.logger.info(f"Loading model {name}")
            model_manager.load_model()
            
        return model_manager
        
    def activate_model(self, name: str, unload_others: bool = True):
        """Activate a model and optionally unload others"""
        if name not in self.models:
            raise ValueError(f"Model {name} not in pool")
            
        # Unload other models if requested
        if unload_others:
            for model_name, manager in self.models.items():
                if model_name != name and manager.model is not None:
                    self.logger.info(f"Unloading model {model_name}")
                    manager.unload_model()
                    
        # Load the requested model
        model_manager = self.models[name]
        if model_manager.model is None:
            model_manager.load_model()
            
        self.active_model = name
        self.logger.info(f"Activated model {name}")
        
    def cleanup(self):
        """Unload all models and clean up"""
        for name, manager in self.models.items():
            manager.unload_model()
        self.models.clear()
        self.active_model = None
