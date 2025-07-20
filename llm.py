import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import gc
from contextlib import contextmanager

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global pipeline to avoid repeated model loading
_pipeline = None
_tokenizer = None
_model = None

def model_loading():
    model_path = "/home/joeloommen/Documents/projects/KSUM_HackgenAI/Models"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        local_files_only=True,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model

def model_init():
    global _pipeline, _tokenizer, _model
    
    if _pipeline is None:
        _tokenizer, _model = model_loading()
        _pipeline = pipeline(
            "text-generation", 
            model=_model, 
            tokenizer=_tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    return _tokenizer, _model

@contextmanager
def cuda_memory_context():
    """Context manager for CUDA memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

SYSTEM_PROMPT = 'You are a helpful assistant.'
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_INST, E_INST = "[INST]", "[/INST]"

def generate_response(llm_model, llm_tokenizer, prompt_user, prompt_system=SYSTEM_PROMPT, temperature=0.4, max_tokens=100):
    global _pipeline
    
    with cuda_memory_context():
        prompt = f"{B_INST} {B_SYS}{prompt_system.strip()}{E_SYS}{prompt_user.strip()} {E_INST}\n\n"
        
        # Use provided model/tokenizer parameters if pipeline not initialized
        if _pipeline is None:
            _pipeline = pipeline(
                "text-generation", 
                model=llm_model, 
                tokenizer=llm_tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        try:
            # Generate with all original parameters maintained
            output = _pipeline(
                prompt, 
                max_new_tokens=max_tokens, 
                do_sample=True, 
                temperature=temperature,
                pad_token_id=llm_tokenizer.eos_token_id
            )
            
            # Extract generated text (maintain original logic)
            full_text = output[0]["generated_text"]
            generated_text = full_text[len(prompt):].strip()
                
            return generated_text
            
        except torch.cuda.OutOfMemoryError:
            # Emergency cleanup and retry
            torch.cuda.empty_cache()
            gc.collect()
            
            # Retry with conservative settings
            output = _pipeline(
                prompt, 
                max_new_tokens=min(max_tokens, 50), 
                do_sample=False,
                pad_token_id=llm_tokenizer.eos_token_id
            )
            
            full_text = output[0]["generated_text"]
            generated_text = full_text[len(prompt):].strip()
                
            return generated_text

def cleanup_model():
    """Explicitly cleanup model resources"""
    global _pipeline, _tokenizer, _model
    
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
    
    if _model is not None:
        del _model
        _model = None
        
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()