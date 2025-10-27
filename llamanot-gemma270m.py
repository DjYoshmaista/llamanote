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
from config import PREPROCESS_PROMPT as preprompt, CHUNK_SIZE as chunk_sz, DEFAULT_MODEL as dflt_mdl

warnings.filterwarnings('ignore')

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

#   except PyPDF2.PdfReadError:
    # print("Error: Invalid or corrupted PDF file")
    # return None
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
        word_length = len(word) + 1 # +1 for the space
        if current_length + word_length > target_chunk_size and current_chunk:
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
        print(f"Error extracting metadta: {str(e)}")
        return None

def format_chat_prompt(system_message: str, user_message: str, tokenizer) -> str:
    """
    Format a chat prompt with fallback for models without chat templates.
    
    This function tries to use the tokenizer's chat template if available,
    otherwise falls back to manual formatting that works with most models.
    """
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    
    # Try to use the tokenizer's chat template
    try:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
    except (ValueError, AttributeError) as e:
        # If chat template doesn't exist or fails, use manual formatting
        pass
    
    # Fallback: Manual formatting that works with most models
    # This format is compatible with Gemma, Llama, Qwen, and many others
    prompt = f"{system_message}\n\n{user_message}\n\n"
    
    # Alternative format for models that prefer explicit instruction markers
    # Uncomment if the above doesn't work well:
    # prompt = f"### System:\n{system_message}\n\n### User:\n{user_message}\n\n### Assistant:\n"
    
    return prompt

def set_text(input_pdf):
    text = ''
    pdf_reader = PdfReader(input_pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Use Llama-3.2-1B-Instruct to extract relevant information
    model = AutoModelForCausalLM.from_pretrained('Llama-3.2-1B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained('Llama-3.2-1B-Instruct')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model.generate(**inputs)
    return outputs

def transcribe(text):
    # Load podcast transcript
    with open(text, 'r') as f:
        text = f.read()

    # Use Qwen3-4B-Instruct to generate the podcast transcript
    transcript_model = AutoModelForCausalLM.from_pretrained('Qwen3-4B-Instruct-2507')
    transcript_tokenizer = AutoTokenizer.from_pretrained('Qwen3-4B-Instruct-2507')
    transcript_inputs = tokenizer(text, return_tensors='pt')
    transcript_outputs = model.generate(**inputs)
    enhanced_transcript = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return enhanced_transcript

def enhance_transcript(transcriptv1):
    with open(transcriptv1, 'r') as f:
        transcript = f.read()

    # Use tertiary model to enhance the transcript
    finalization_model = AutoModelForCausalLM.from_pretrained('Qwen3-4B-Instruct-2507')
    finalization_tokenizer = AutoTokenizer.from_pretrained('Qwen3-4B-Instruct-2507')
    inputs = tokenizer(transcript, return_tensors='pt')
    outputs = model.generate(**inputs)
    enhanced_transcript = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return enhanced_transcript

def podcast_gen(enhanced_transcript):
    with open(enhanced_transcript, 'r') as f:
        transcript = f.read()

    # Use TTS model to generate the podcast
    model = AutoModelForTextToWaveform.from_pretrained('parler-tts/parler-tts-mini-v1')
    processor = AutoProcessor.from_pretrained('parler-tts/parler-tts-mini-v1')
    inputs = processor(transcript, return_tensors='pt')
    outputs = model.generate(**inputs)
    return outputs

def prepare_model(device, use_quantization=True, quantization_type="8bit", max_memory_per_gpu="10GB"):
    """
    Prepare model with multiple VRAM optimization strategies.
    
    Args:
        device: Target device ('cuda' or 'cpu')
        use_quantization: Whether to use quantization (reduces VRAM significantly)
        quantization_type: Either '8bit' or '4bit' (4bit uses even less memory)
        max_memory_per_gpu: Maximum VRAM to use per GPU (e.g., "10GB", "8GB")
    
    VRAM Usage Estimates for Qwen3-4B:
    - No quantization: ~16GB
    - 8-bit quantization: ~8GB  (50% reduction)
    - 4-bit quantization: ~4GB  (75% reduction)
    - With CPU offloading: Can run with <4GB VRAM
    """
    
    print(f"\n{'='*60}")
    print("MEMORY OPTIMIZATION SETTINGS:")
    print(f"  Device: {device}")
    print(f"  Quantization: {use_quantization} ({quantization_type if use_quantization else 'None'})")
    print(f"  Max VRAM per GPU: {max_memory_per_gpu}")
    print(f"{'='*60}\n")
    
    if use_quantization and device == "cuda":
        try:
            # OPTION 1: 8-bit quantization (recommended - best balance of speed/memory)
            if quantization_type == "8bit":
                print("Loading model with 8-bit quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    dflt_mdl,
                    load_in_8bit=True,  # Reduces memory by ~50%
                    device_map="auto",  # Automatically splits across GPU/CPU
                    max_memory={0: max_memory_per_gpu},  # Limit VRAM usage
                    torch_dtype=torch.bfloat16,
                )
                
            # OPTION 2: 4-bit quantization (maximum memory savings, slightly slower)
            elif quantization_type == "4bit":
                print("Loading model with 4-bit quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    dflt_mdl,
                    load_in_4bit=True,  # Reduces memory by ~75%
                    device_map="auto",
                    max_memory={0: max_memory_per_gpu},
                    torch_dtype=torch.bfloat16,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,  # Even more memory efficient
                    bnb_4bit_quant_type="nf4"  # Normalized float 4-bit
                )
            
            tokenizer = AutoTokenizer.from_pretrained(dflt_mdl)
            
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
        accelerator = Accelerator()
        
        model = AutoModelForCausalLM.from_pretrained(
            dflt_mdl,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            device_map="auto",  # Automatically distribute layers between GPU and CPU
            max_memory={
                0: max_memory_per_gpu,  # Limit GPU memory
                "cpu": "30GB"  # Use up to 30GB of system RAM
            },
            low_cpu_mem_usage=True,  # Reduces peak memory during loading
            offload_folder="offload",  # Folder for offloading weights
        )
        
        tokenizer = AutoTokenizer.from_pretrained(dflt_mdl, use_safetensors=True)
        
        print(f"\n✓ Model loaded with device map: {model.hf_device_map}")
        print("✓ Layers are split between GPU VRAM and system RAM")
        
        return model, tokenizer

def process_chunk(model, tokenizer, device, text_chunk, chunk_num):
    """
    Process a chunk of text and return the processed output.
    
    This version uses format_chat_prompt() which handles models
    without chat templates (like Gemma).
    """
    # Format the prompt using our flexible formatting function
    prompt = format_chat_prompt(preprompt, text_chunk, tokenizer)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the correct device
    # Note: When using device_map="auto", the model handles device placement automatically
    # But we still need to move inputs to the same device as the model's first layer
    if hasattr(model, 'hf_device_map'):
        # Get the device of the first model parameter
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    # Decode the output, removing the prompt
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Try to extract just the response (after the prompt)
    if full_output.startswith(prompt):
        processed_text = full_output[len(prompt):].strip()
    else:
        # Fallback: try to find where the actual response starts
        processed_text = full_output.strip()

    # Print chunk information for monitoring
    print(f"\n{'='*40} Chunk {chunk_num} {'='*40}")
    print(f"INPUT TEXT:\n{text_chunk[:200]}...") # Show first 200 chars of input
    print(f"\nPROCESSED TEXT:\n{processed_text[:200]}...") # Show first 200 chars of output
    print(f"{'='*90}\n")

    return processed_text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # PDF extraction phase
    while True:
        input_pdf = Path(input("Input the filepath to the pdf file you would like to convert to text: "))
        pdf_meta = get_pdf_metadata(input_pdf)
        if pdf_meta:
            print("PDF Metadata retrieved!  Metadata:\n")
            print("==="*15)
            for key, value in pdf_meta['metadata'].items():
                print(f"{key}: {value}")
            print("==="*15)
            pdf_text = extract_text_from_pdf(input_pdf)
            print(f"PDF Text Extracted!  First 500 characters of PDF Text:\n{"==="*15}\n{str(pdf_text)[:500]}\n{"==="*15}")
            print(f"Total number of characters extracted: {len(pdf_text)}")
        else:
            print("=== PDF Metadata not retrieved ===")
            continue
        if pdf_text:
            output_file = str(os.path.basename(input_pdf))
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(pdf_text)
            print(f"\nExtracted text has been saved to {output_file}")
        break
    
    # MEMORY OPTIMIZATION CONFIGURATION
    print("\n" + "="*60)
    print("VRAM OPTIMIZATION OPTIONS:")
    print("="*60)
    print("1. 8-bit quantization (Recommended - ~8GB VRAM, best balance)")
    print("2. 4-bit quantization (Maximum savings - ~4GB VRAM, slightly slower)")
    print("3. Auto device mapping (Splits layers between GPU/CPU)")
    print("4. No optimization (Uses ~16GB VRAM)")
    print("="*60)
    
    optimization_choice = input("\nSelect optimization (1-4, default=1): ").strip() or "1"
    
    if optimization_choice == "1":
        model, tokenizer = prepare_model(device, use_quantization=True, quantization_type="8bit")
    elif optimization_choice == "2":
        model, tokenizer = prepare_model(device, use_quantization=True, quantization_type="4bit")
    elif optimization_choice == "3":
        max_vram = input("Max VRAM to use (e.g., '10GB', '8GB', default='10GB'): ").strip() or "10GB"
        model, tokenizer = prepare_model(device, use_quantization=False, max_memory_per_gpu=max_vram)
    else:
        # Original method - no optimization
        accelerator = Accelerator()
        model = AutoModelForCausalLM.from_pretrained(
            dflt_mdl,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(dflt_mdl, use_safetensors=True)
        model, tokenizer = accelerator.prepare(model, tokenizer)
    
    # Check if tokenizer has padding token, if not set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"⚠ Note: Model doesn't have a padding token. Using EOS token as pad token.")
    
    # Chunk size configuration
    target_chunk_sz = input("What would you like for the target chunk size to be?  Default value is 1000, target chunk size set to default value after 5 incorrect input attempts.  Range is 5-5000: ")
    n = 5
    while n > 0:
        if not target_chunk_sz.isdigit():
            target_chunk_sz = input(f"ERROR: Target chunk size not set as an integer!  Please reenter a valid integer for the target chunk size: ")
            if target_chunk_sz.isdigit() and int(target_chunk_sz) >= 5 and int(target_chunk_sz) <= 5000:
                break
        elif int(target_chunk_sz) < 5 or int(target_chunk_sz) > 5000:
            target_chunk_sz = input(f"ERROR: Target chunk size is out of range!  Please reenter a valid integer in range 5-5000: ")
            if target_chunk_sz.isdigit() and int(target_chunk_sz) >= 5 and int(target_chunk_sz) <= 5000:
                break
        else:
            break
        n -= 1
    
    if not target_chunk_sz.isdigit() or int(target_chunk_sz) < 5 or int(target_chunk_sz) > 5000:
        target_chunk_sz = 1000
        print(f"Setting target chunk size to default: {target_chunk_sz}")
    else:
        target_chunk_sz = int(target_chunk_sz)
    
    # Once the target chunk size has been set, read contents of the textual output from the pdf
    with open(output_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Create chunks from output file
    chunks = create_word_bounded_chunks(text, target_chunk_sz)

    # Calculate number of chunks
    num_chunks = len(chunks)
    print(f"\nProcessing {num_chunks} chunks...")

    # Create output file name
    chunked_file = f"clean_{os.path.basename(output_file)}"
    
    processed_text = ""

    with open(chunked_file, 'w', encoding='utf-8') as chunked_out_file:
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Process chunk and append to complete text
            processed_chunk = process_chunk(model, tokenizer, device, chunk, chunk_num)
            processed_text += processed_chunk + "\n"

            # Write chunk immediately to file
            chunked_out_file.write(processed_chunk + "\n")
            chunked_out_file.flush()
    
    print(f"\n✓ Processing complete! Output saved to: {chunked_file}")


"""
    text = set_text(pdf_text)
    transcriptv1 = transcribe(text)
    final_transcript = enhance_transcript(transcriptv1)
    audio = podcast_gen(final_transcript)
"""

if __name__ == "__main__":
    main()
