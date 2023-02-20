import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from photo_gpt.tools.image_xray import get_mask_image


class MaskFormer:
    def __init__(self, image_store: dict = None):
        self.image_store = image_store

    def get_mask_from_text(
        self, image_path, text, threshold=0.5, padding=20, min_area=0.02
    ):
        image = Image.open(image_path)
        image = image.resize((512, 512))
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )

        inputs = processor(
            text=text,
            images=image,
            padding="max_length",
            return_tensors="pt",
        )

        # predict
        with torch.no_grad():
            outputs = model(**inputs)

        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold

        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])

        if area_ratio < min_area:
            return None

        image_mask = get_mask_image(mask, padding=padding)

        return image_mask.resize(image.size)
