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
        self.user_prompt = "find potential keywords to search for news articles and relevant articles that can be found in the internet from the following text: " + prompt
        