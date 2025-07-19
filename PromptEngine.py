import os

class PromptEngine:
    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def extract_keyword(self, prompt):
        self.user_prompt = "find potential keywords to search for news articles and relevant articles that can be found in the internet from the following text: " + prompt
        inputs = self.tokenizer(self.user_prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=100)
        return outputs.text[0].decode('utf-8')