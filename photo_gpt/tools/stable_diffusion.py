import uuid

import torch
from diffusers import StableDiffusionPipeline


from photo_gpt.tools.telegram_utls import TelegramHelper


class StableDiffusion:
    def __init__(self, telegram_helper: TelegramHelper, image_store: dict = None):
        self.image_store = image_store
        self.telegram_helper = telegram_helper

    def run(self, text_to_image: str) -> str:
        """Run query through CLIP and parse result."""

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe = pipe.to("mps")

        # Recommended if your computer has < 64 GB of RAM
        pipe.enable_attention_slicing()

        print(f"Image with text {text_to_image} is being created...")

        image = pipe(text_to_image).images[0]

        image_path = str(uuid.uuid4())[0:8] + ".png"

        image.save(image_path)

        output = f"An image of {text_to_image} has been created and saved to path: {image_path}\n"

        output += self.telegram_helper.send_photo_to_user(image_path)
        print(output)
        return output
