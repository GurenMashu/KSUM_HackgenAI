import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import llm

tokenizer, model = llm.model_loading()

class PromptEngine:
    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def extract_keyword(self, prompt):
        self.user_prompt = prompt
        self.system_prompt = "find potential keywords to search for news articles and relevant articles that can be found in the internet from the following text: "
        self.generated_text = llm.generate_response(llm_model=self.model, llm_tokenizer=self.tokenizer, prompt_user=self.user_prompt, prompt_system=self.system_prompt)
        print(self.generated_text)

prompt_engine = PromptEngine(model=model, tokenizer=tokenizer)
prompt_engine.extract_keyword(prompt="Hello, how are you my good man?")