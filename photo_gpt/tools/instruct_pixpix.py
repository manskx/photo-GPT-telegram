import json
import uuid


import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
from PIL import Image

from photo_gpt.tools.telegram_utls import TelegramHelper


class Pix2Pix:
    def __init__(self, telegram_helper: TelegramHelper, image_store: dict = None):
        self.image_store = image_store
        self.telegram_helper = telegram_helper

    def change_style_of_image(self, input_json):
        """Change style of image."""

        parsed_json = json.loads(input_json.strip().replace("'", '"'))
        image_path = parsed_json["image_path"]
        instruct_text = parsed_json["instruct_text"]

        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        )
        pipe = pipe.to("mps")

        # Recommended if your computer has < 64 GB of RAM
        pipe.enable_attention_slicing()

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        original_image = Image.open(image_path)

        image = pipe(
            instruct_text,
            image=original_image,
            num_inference_steps=40,
            image_guidance_scale=1.2,
        ).images[0]

        updated_image_path = "{}_updated_{}.png".format(
            image_path.split(".")[0], str(uuid.uuid4())[0:4]
        )

        image.save(updated_image_path)

        output = f"Style {instruct_text} has been applied to image {image_path} and saved to path: {updated_image_path}\n"

        output += self.telegram_helper.send_photo_to_user(updated_image_path)
        print(output)

        return output
