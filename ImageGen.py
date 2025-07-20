import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

import os
from huggingface_hub import InferenceClient

class ImageGen:
    def __init__(self, provider, api_key, model):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.client = InferenceClient(
            provider=self.provider,
            api_key=self.api_key,
        )
    def text_to_image(self, prompt):
        return self.client.text_to_image(
            prompt,
            model=self.model,
        )

# Example usage
if __name__ == "__main__":
    provider = "nebius"
    api_key = os.getenv("HF_TOKEN")
    model = "black-forest-labs/FLUX.1-dev"

    image_gen = ImageGen(provider, api_key, model)
    prompt = "A futuristic cityscape at sunset"
    image = image_gen.text_to_image(prompt)

    image.save("output_image.jpg")