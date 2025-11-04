# llamanote.py (updated with integrated functions and cloud support)
import transformers
import pickle
from PyPDF2 import PdfReader
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTextToWaveform, AutoProcessor
import PyPDF2
from typing import Optional
import os
import torch
from accelerate import Accelerator
from tqdm import tqdm
import warnings
import requests
import json

from config import PREPROCESS_PROMPT as preprompt, CHUNK_SIZE as chunk_sz
from config_manager import ConfigManager

warnings.filterwarnings('ignore')

class CloudModelClient:
    """Client for making API calls to cloud providers"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.cloud_config = config_manager.current_config["cloud_settings"]
    
    def generate_with_cloud(self, prompt: str, step: str, system_prompt: str = None) -> str:
        """Generate text using cloud provider"""
        step_config = self.config_manager.get_model_config(step)
        provider = step_config["provider"]
        model_name = step_config["model_name"]
        
        if provider == "openai":
            return self._call_openai(prompt, model_name, system_prompt)
        elif provider == "anthropic":
            return self._call_anthropic(prompt, model_name, system_prompt)
        elif provider == "google":
            return self._call_google(prompt, model_name, system_prompt)
        elif provider == "mistral":
            return self._call_mistral(prompt, model_name, system_prompt)
        elif provider == "deepseek":
            return self._call_deepseek(prompt, model_name, system_prompt)
        elif provider == "openrouter":
            return self._call_openrouter(prompt, model_name, system_prompt)
        elif provider == "qwen":
            return self._call_qwen(prompt, model_name, system_prompt)
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")
    
    def _call_openai(self, prompt: str, model: str, system_prompt: str = None) -> str:
        """Call OpenAI API"""
        api_key = self.cloud_config.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key not configured")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API error: {response.text}")
    
    def _call_anthropic(self, prompt: str, model: str, system_prompt: str = None) -> str:
        """Call Anthropic Claude API"""
        api_key = self.cloud_config.get("anthropic_api_key")
        if not api_key:
            raise ValueError("Anthropic API key not configured")
        
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            raise Exception(f"Anthropic API error: {response.text}")
    
    def _call_google(self, prompt: str, model: str, system_prompt: str = None) -> str:
        """Call Google Gemini API"""
        api_key = self.cloud_config.get("google_api_key")
        if not api_key:
            raise ValueError("Google API key not configured")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(full_prompt)
            return response.text
        except ImportError:
            raise Exception("Google Generative AI package not installed. Run: pip install google-generativeai")
    
    def _call_mistral(self, prompt: str, model: str, system_prompt: str = None) -> str:
        """Call Mistral API"""
        api_key = self.cloud_config.get("mistral_api_key")
        if not api_key:
            raise ValueError("Mistral API key not configured")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Mistral API error: {response.text}")
    
    def _call_deepseek(self, prompt: str, model: str, system_prompt: str = None) -> str:
        """Call DeepSeek API"""
        api_key = self.cloud_config.get("deepseek_api_key")
        if not api_key:
            raise ValueError("DeepSeek API key not configured")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"DeepSeek API error: {response.text}")
    
    def _call_openrouter(self, prompt: str, model: str, system_prompt: str = None) -> str:
        """Call OpenRouter API"""
        api_key = self.cloud_config.get("openrouter_api_key")
        if not api_key:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-username/llamanote",
            "X-Title": "LlamaNote Podcast Generator"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error: {response.text}")
    
    def _call_qwen(self, prompt: str, model: str, system_prompt: str = None) -> str:
        """Call Qwen API"""
        api_key = self.cloud_config.get("qwen_api_key")
        if not api_key:
            raise ValueError("Qwen API key not configured")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,
            "input": {"messages": messages},
            "parameters": {
                "max_tokens": 512,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["output"]["text"]
        else:
            raise Exception(f"Qwen API error: {response.text}")

# PDF Processing Functions (updated to work with new system)
def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    if not str(file_path).lower().endswith('.pdf'):
        print("Error: File does not have '.pdf' extension")
        return False
    return True

def extract_text_from_pdf(file_path: str, max_chars: int = 100000000) -> Optional[str]:
    if not validate_pdf(file_path):
        return None

    try:
        with open(file_path, 'rb') as file:
            # Create PDF Reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")

            extracted_text = []
            total_chars = 0

            # Iterate through all pages
            for page_num in range(num_pages):
                # Extract text from page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Check if adding this page's text would exceed the limit
                if total_chars + len(text) > max_chars:
                    # Only add text up to the limit
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break

                extracted_text.append(text)
                total_chars += len(text)
                print(f"Processed page {page_num + 1}/{num_pages}")

            final_text = '\n'.join(extracted_text)
            print(f"\nExtraction complete!  Total characters: {len(final_text)}")
            return final_text

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

def create_word_bounded_chunks(text, target_chunk_size):
    # Split text into chunks at word boundaries close to the target chunk size
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if int(current_length) + int(word_length) > int(target_chunk_size) and len(current_chunk):
            # Join the current chunk and add it to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_pdf_metadata(file_path: str) -> Optional[dict]:
    if not validate_pdf(file_path):
        return None

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'metadata': pdf_reader.metadata
            }
            return metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return None

# Model Preparation Function (updated to use config)
def prepare_model(device, model_name, use_quantization=True, quantization_type="8bit", max_memory_per_gpu="10GB"):
    """
    Prepare model with multiple VRAM optimization strategies.
    """
    
    print(f"\n{'='*60}")
    print("MEMORY OPTIMIZATION SETTINGS:")
    print(f"  Device: {device}")
    print(f"  Model: {model_name}")
    print(f"  Quantization: {use_quantization} ({quantization_type if use_quantization else 'None'})")
    print(f"  Max VRAM per GPU: {max_memory_per_gpu}")
    print(f"{'='*60}\n")
    
    if use_quantization and device == "cuda":
        try:
            # OPTION 1: 8-bit quantization
            if quantization_type == "8bit":
                print("Loading model with 8-bit quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    max_memory={0: max_memory_per_gpu},
                    torch_dtype=torch.bfloat16,
                )
                
            # OPTION 2: 4-bit quantization
            elif quantization_type == "4bit":
                print("Loading model with 4-bit quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_4bit=True,
                    device_map="auto",
                    max_memory={0: max_memory_per_gpu},
                    torch_dtype=torch.bfloat16,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"\n✓ Model loaded successfully with {quantization_type} quantization")
            print(f"✓ Model device map: {model.hf_device_map}")
            return model, tokenizer
            
        except Exception as e:
            print(f"\n⚠ Warning: Quantization failed ({str(e)})")
            print("Falling back to standard loading with CPU offloading...")
            use_quantization = False
    
    # OPTION 3: No quantization but with automatic GPU/CPU layer splitting
    if not use_quantization:
        print("Loading model with automatic device mapping (GPU/CPU split)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            device_map="auto",
            max_memory={
                0: max_memory_per_gpu,
                "cpu": "30GB"
            },
            low_cpu_mem_usage=True,
            offload_folder="offload",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
        
        print(f"\n✓ Model loaded with device map: {model.hf_device_map}")
        print("✓ Layers are split between GPU VRAM and system RAM")
        
        return model, tokenizer

# Processing Functions (updated to use configuration system)
def process_chunk_with_config(config_manager: ConfigManager, cloud_client: CloudModelClient, 
                             text_chunk: str, chunk_num: int, step: str = "preprocessing") -> str:
    """
    Process a chunk of text using either local or cloud model based on configuration
    """
    step_config = config_manager.get_model_config(step)
    
    if step_config["model_type"] == "cloud":
        # Use cloud model
        print(f"Processing {step} chunk {chunk_num} with cloud model: {step_config['provider']}/{step_config['model_name']}")
        try:
            system_prompt = preprompt if step == "preprocessing" else None
            processed_text = cloud_client.generate_with_cloud(text_chunk, step, system_prompt)
            
            # Print monitoring information
            print(f"\n{'='*40} {step.title()} Chunk {chunk_num} {'='*40}")
            print(f"INPUT TEXT:\n{text_chunk[:200]}...")
            print(f"PROCESSED TEXT:\n{processed_text[:200]}...")
            print(f"{'='*90}\n")
            
            return processed_text
        except Exception as e:
            print(f"Cloud API error: {str(e)}")
            print("Falling back to default local model...")
            # Update config to use local model as fallback
            step_config["model_type"] = "local"
            step_config["provider"] = "huggingface"
            step_config["model_name"] = "google/gemma-3-270m"  # Default fallback
    
    # Use local model
    print(f"Processing {step} chunk {chunk_num} with local model: {step_config['model_name']}")
    
    # This would need to be implemented based on your local model setup
    # For now, we'll return a placeholder
    return f"Processed chunk {chunk_num} with local model (implementation needed)"

def transcribe_with_config(config_manager: ConfigManager, cloud_client: CloudModelClient, text: str) -> str:
    """Transcribe text using configured model"""
    return process_chunk_with_config(config_manager, cloud_client, text, 0, "transcription")

def enhance_with_config(config_manager: ConfigManager, cloud_client: CloudModelClient, transcript: str) -> str:
    """Enhance transcript using configured model"""
    return process_chunk_with_config(config_manager, cloud_client, transcript, 0, "enhancement")

def text_to_speech_with_config(config_manager: ConfigManager, text: str) -> any:
    """Convert text to speech using configured model"""
    tts_config = config_manager.get_model_config("text_to_speech")
    
    if tts_config["model_type"] == "local":
        try:
            model = AutoModelForTextToWaveform.from_pretrained(tts_config["model_name"])
            processor = AutoProcessor.from_pretrained(tts_config["model_name"])
            inputs = processor(text, return_tensors='pt')
            outputs = model.generate(**inputs)
            return outputs
        except Exception as e:
            print(f"TTS model error: {str(e)}")
            return None
    else:
        # Cloud TTS would go here (e.g., OpenAI TTS, Google TTS, etc.)
        print("Cloud TTS not yet implemented")
        return None

def read_file_to_string(filename):
    # Try UTF-8 first
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        # If UTF-8 fails, try with other common encodings
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"Successfully read file using {encoding} encoding.")
                return content
            except UnicodeDecodeError:
                continue

        print(f"Error: Could not decode file '{filename}' with any common encoding.")
        return None
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except IOError:
        print(f"Error: Could not read file '{filename}'.")
        return None

# Updated Main Function
def main():
    # Initialize configuration manager
    config_manager = ConfigManager()
    cloud_client = None
    
    print("="*60)
    print("LLAMANOTE - PODCAST GENERATION SYSTEM")
    print("="*60)
    
    # Configuration management menu
    while True:
        print("\nConfiguration Management:")
        print("1. Load existing configuration")
        print("2. Create new configuration")
        print("3. List available configurations")
        print("4. Delete configuration")
        print("5. Update configuration directory")
        print("6. Continue with current configuration")
        
        config_choice = input("\nSelect option (1-6, default=6): ").strip() or "6"
        
        if config_choice == "1":
            configs = config_manager.list_configs()
            if configs:
                print("\nAvailable configurations:")
                for i, config in enumerate(configs, 1):
                    print(f"{i}. {config}")
                try:
                    choice = int(input("Select configuration: ").strip())
                    if 1 <= choice <= len(configs):
                        config_manager.load_config(configs[choice-1])
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            else:
                print("No configurations found.")
                
        elif config_choice == "2":
            config_name = input("Enter name for new configuration: ").strip()
            if config_name:
                new_config = config_manager.interactive_config_creation()
                config_manager.create_config(config_name, new_config)
                
        elif config_choice == "3":
            configs = config_manager.list_configs()
            if configs:
                print("\nAvailable configurations:")
                for config in configs:
                    print(f"- {config}")
            else:
                print("No configurations found.")
                
        elif config_choice == "4":
            configs = config_manager.list_configs()
            if configs:
                print("\nAvailable configurations:")
                for i, config in enumerate(configs, 1):
                    print(f"{i}. {config}")
                try:
                    choice = int(input("Select configuration to delete: ").strip())
                    if 1 <= choice <= len(configs):
                        config_manager.delete_config(configs[choice-1])
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            else:
                print("No configurations found.")
                
        elif config_choice == "5":
            new_path = input("Enter new configuration directory path: ").strip()
            if new_path:
                config_manager.update_config_dir(new_path)
                
        elif config_choice == "6":
            # Load default config if none loaded
            if not config_manager.current_config:
                config_manager.current_config = config_manager.default_config
            break
    
    # Initialize cloud client if any step uses cloud models
    for step in ["preprocessing", "transcription", "enhancement"]:
        step_config = config_manager.get_model_config(step)
        if step_config["model_type"] == "cloud":
            cloud_client = CloudModelClient(config_manager)
            break
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # PDF extraction phase
    while True:
        input_pdf = input("Input the filepath to the pdf file you would like to convert to text: ").strip()
        if not input_pdf:
            print("No file path provided. Please try again.")
            continue
            
        input_pdf = Path(input_pdf)
        pdf_meta = get_pdf_metadata(input_pdf)
        
        if pdf_meta:
            print("PDF Metadata retrieved!  Metadata:\n")
            print("==="*15)
            for key, value in pdf_meta['metadata'].items():
                print(f"{key}: {value}")
            print("==="*15)
            
            pdf_text = extract_text_from_pdf(input_pdf)
            if pdf_text:
                print(f"PDF Text Extracted!  First 500 characters of PDF Text:\n{'==='*15}\n{str(pdf_text)[:500]}\n{'==='*15}")
                print(f"Total number of characters extracted: {len(pdf_text)}")
                
                output_file = str(os.path.basename(input_pdf)) + "_extracted.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(pdf_text)
                print(f"\nExtracted text has been saved to {output_file}")
                break
            else:
                print("Failed to extract text from PDF. Please try another file.")
        else:
            print("=== PDF Metadata not retrieved ===")
            continue
    
    # Model preparation - only prepare local models if needed for preprocessing
    preprocessing_config = config_manager.get_model_config("preprocessing")
    local_model = None
    local_tokenizer = None
    
    if preprocessing_config["model_type"] == "local":
        print(f"\nPreparing local model for preprocessing: {preprocessing_config['model_name']}")
        
        # MEMORY OPTIMIZATION CONFIGURATION
        print("\n" + "="*60)
        print("VRAM OPTIMIZATION OPTIONS:")
        print("="*60)
        print("1. 8-bit quantization (Recommended - best balance)")
        print("2. 4-bit quantization (Maximum savings - slightly slower)")
        print("3. Auto device mapping (Splits layers between GPU/CPU)")
        print("4. No optimization (Uses full VRAM)")
        print("="*60)
        
        optimization_choice = input("\nSelect optimization (1-4, default=1): ").strip() or "1"
        max_vram = "10GB"
        
        if optimization_choice == "1":
            local_model, local_tokenizer = prepare_model(
                device, preprocessing_config["model_name"], 
                use_quantization=True, quantization_type="8bit", max_memory_per_gpu=max_vram
            )
        elif optimization_choice == "2":
            local_model, local_tokenizer = prepare_model(
                device, preprocessing_config["model_name"],
                use_quantization=True, quantization_type="4bit", max_memory_per_gpu=max_vram
            )
        elif optimization_choice == "3":
            max_vram = input("Max VRAM to use (e.g., '10GB', '8GB', default='10GB'): ").strip() or "10GB"
            local_model, local_tokenizer = prepare_model(
                device, preprocessing_config["model_name"],
                use_quantization=False, max_memory_per_gpu=max_vram
            )
        else:
            # No optimization
            accelerator = Accelerator()
            local_model = AutoModelForCausalLM.from_pretrained(
                preprocessing_config["model_name"],
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                device_map=device,
            )
            local_tokenizer = AutoTokenizer.from_pretrained(preprocessing_config["model_name"], use_safetensors=True)
            local_model, local_tokenizer = accelerator.prepare(local_model, local_tokenizer)
    
    # Chunk processing
    target_chunk_sz = input("What would you like for the target chunk size to be? Default is 1000, range 5-5000: ")
    n = 5
    while n > 0:
        if not target_chunk_sz.isdigit():
            target_chunk_sz = input("ERROR: Target chunk size not set as an integer! Please reenter a valid integer: ")
            if target_chunk_sz.isdigit() and 5 <= int(target_chunk_sz) <= 5000:
                break
        elif int(target_chunk_sz) < 5 or int(target_chunk_sz) > 5000:
            target_chunk_sz = input("ERROR: Target chunk size is out of range! Please reenter a valid integer in range 5-5000: ")
            if target_chunk_sz.isdigit() and 5 <= int(target_chunk_sz) <= 5000:
                break
        else:
            break
        n -= 1
    
    if not target_chunk_sz.isdigit() or int(target_chunk_sz) < 5 or int(target_chunk_sz) > 5000:
        target_chunk_sz = 1000
        print(f"Setting target chunk size to default: {target_chunk_sz}")
    else:
        target_chunk_sz = int(target_chunk_sz)
    
    # Read extracted text and create chunks
    with open(output_file, 'r', encoding='utf-8') as file:
        text = file.read()

    chunks = create_word_bounded_chunks(text, target_chunk_sz)
    num_chunks = len(chunks)
    print(f"\nProcessing {num_chunks} chunks...")

    # Process chunks using configured model
    cleaned_file = f"cleaned_{os.path.basename(output_file)}"
    
    with open(cleaned_file, 'w', encoding='utf-8') as out_file:
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            processed_chunk = process_chunk_with_config(config_manager, cloud_client, chunk, chunk_num, "preprocessing")
            out_file.write(processed_chunk + "\n")
            out_file.flush()
    
    print(f"\n✓ Preprocessing complete! Output saved to: {cleaned_file}")
    
    # Continue with other processing steps (transcription, enhancement, TTS) as needed
    print("\nWould you like to continue with podcast generation?")
    continue_choice = input("Generate podcast? (y/N): ").strip().lower()
    
    if continue_choice == 'y':
        print("Welcome to the preprocessing stage!")
        print("==="*15)
        user_input = input(f"\n\nEnter the filepath for the file you would like to read in, load, and preprocess the textual data from: ")
        INPUT_PROMPT = read_file_to_string(user_input)
        pipeline = transformers.pipeline("text-generation", model=MODEL, model_kwargs={"torch_dtype": torch.bfloat16}, device_map=device)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": INPUT_PROMPT}]
        outputs = pipeline(messages, max_new_tokens=8126, temperature=1)

        save_string_pkl = outputs[0]["generated_text"][-1]['content']
        print(outputs[0]["generated_text"][-1]['content'])

        save_file = input("Enter the filename for the file you would like to save (the .pkl file)")
        with open(save_file, 'wb') as file:
            pickle.dump(save_string_pkl, file)
        print("Pickle file saved!")
        # This is where you would implement the full pipeline
        # using transcribe_with_config, enhance_with_config, and text_to_speech_with_config
        print("Podcast generation pipeline would continue here...")
        # Implementation would follow the same pattern as preprocessing

if __name__ == "__main__":
    main()
