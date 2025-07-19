import torch
import torch.version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')
print(f'Running torch version {torch.__version__}')

def model_loading():
    model_path = "/home/joeloommen/Documents/projects/KSUM_HackgenAI/Models"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        local_files_only=True,
        quantization_config=quant_config
    )
    
    return tokenizer, model


tokenizer, model = model_loading()

def generate_response(prompt, temperature=0.):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.5)
    return output[0]["generated_text"]