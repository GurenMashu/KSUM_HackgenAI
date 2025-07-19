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

class KeywordEngine:
    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def extract_keyword(self, prompt,):
        self.user_prompt = prompt
        self.system_prompt = """
        You are an intelligent assistant that extracts potential keywords for searching relevant news articles and internet content.

        The input text may include multiple unrelated topics. Your task is to extract **separate keyword phrases for each distinct topic** in the paragraph.

        Focus on named entities, events, and phrases someone might search on Google News or in academic/research databases. Do not include filler words or full sentences. Keep keywords short, relevant, and grouped by topic.

        Format:
        Input: <text>
        Topic 1 Keywords: <comma-separated list>
        Topic 2 Keywords: <comma-separated list>
        ...
        If there's only one topic, just give a single line of keywords.
        """

        # maybe change prompt to have json style output like below.

        """
        You are an intelligent assistant that extracts potential keywords for searching relevant news articles and internet content.

        The input text may contain one or more unrelated topics. Your task is to identify each topic and return a list of relevant keyword phrases **grouped by topic**.

        Return the result in **valid JSON format**. Do not include any extra explanation or commentary—only the JSON object.

        Focus on:
        - Real-world entities, events, or research-related phrases
        - Short, relevant search phrases (no full sentences)
        - Group keywords logically per topic

        Format:
        {
        "Topic 1": ["keyword1", "keyword2", ...],
        "Topic 2": ["keyword1", "keyword2", ...]
        }
        If there's only one topic, just use a single key like:
        {
        "Topic": ["keyword1", "keyword2", ...]
        }
        """

        self.generated_text = llm.generate_response(llm_model=self.model, llm_tokenizer=self.tokenizer, prompt_user=self.user_prompt, prompt_system=self.system_prompt)
        print(self.generated_text)

prompt_engine = KeywordEngine(model=model, tokenizer=tokenizer)
prompt_engine.extract_keyword(prompt="Elon musk went to mars and decided to procreate with an alien there. There, another new child was born, a half human, half martian. Scientists report a total loss of life on earth.")