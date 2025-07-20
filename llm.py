import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Running on {device}')
# print(f'Running torch version {torch.__version__}')



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
        quantization_config=quant_config,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    return tokenizer, model

def model_init():
    tokenizer, model = model_loading()
    return tokenizer, model

# tokenizer, model = model_loading()

SYSTEM_PROMPT = 'You are a helpful assistant.'
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_INST, E_INST = "[INST]", "[/INST]"


def generate_response(llm_model, llm_tokenizer, prompt_user, prompt_system=SYSTEM_PROMPT, temperature=0.4, max_tokens=100):
    prompt = f"{B_INST} {B_SYS}{prompt_system.strip()}{E_SYS}{prompt_user.strip()} {E_INST}\n\n"
    pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)
    output = pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
    full_text = output[0]["generated_text"]
    generated_text = full_text[len(prompt):].strip()
    return generated_text

# user_prompt = 'Howdy!'
# system_prompt = 'You are a helpful assistant.'

# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# B_INST, E_INST = "[INST]", "[/INST]"

# prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"
# response = generate_response(prompt)
# print(response)