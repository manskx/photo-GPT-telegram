import logging
import uuid

import torch
from diffusers import StableDiffusionPipeline


from photo_gpt.tools.telegram_utls import TelegramHelper


class StableDiffusion:
    def __init__(self, telegram_helper: TelegramHelper, image_store: dict = None):
        self.image_store = image_store
        self.telegram_helper = telegram_helper

    def run(self, prompt: str) -> str:
        """Run query through CLIP and parse result."""

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )

        pipe = pipe.to("mps")  # or cuda or cpu

        # Recommended if your computer has < 64 GB of RAM
        pipe.enable_attention_slicing()

        logging.info(f"Image with text '{prompt}' is being created...")
        image = pipe(prompt).images[0]
        image_path = str(uuid.uuid4())[0:8] + ".png"
        image.save(image_path)
        output = f"An image of '{prompt}' has been created and saved to path: '{image_path}'\n"

        output += self.telegram_helper.send_photo_to_user(image_path)
        logging.info(output)
        return output
