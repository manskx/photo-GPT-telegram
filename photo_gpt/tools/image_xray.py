import json
import math
import uuid
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)


def get_image_contents_description_1(image_path):
    image_to_text = pipeline(
        "image-to-text", model="nlpconnect/vit-gpt2-image-captioning"
    )
    text_description = image_to_text(image_path)
    return text_description[0]["generated_text"]


def get_image_contents_description_2(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    raw_image = Image.open(image_path)

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def _get_mask_segment_meta(mask_array):
    # get information like area, location, width, height, and approximate center of the mask and approximate location
    # of the approximate center of the mask in the format [up/down, left/right]
    mask_info = {}
    mask_info["area"] = np.count_nonzero(mask_array)
    mask_info["area_percent"] = mask_info["area"] / mask_array.size
    mask_info["location"] = np.where(mask_array == 1)
    mask_info["width"] = mask_info["location"][1].max() - mask_info["location"][1].min()
    mask_info["height"] = (
        mask_info["location"][0].max() - mask_info["location"][0].min()
    )
    mask_info["center"] = [
        mask_info["location"][0].mean(),
        mask_info["location"][1].mean(),
    ]
    mask_info["center_location"] = [
        "up" if mask_info["center"][0] < mask_array.shape[0] / 2 else "down",
        "left" if mask_info["center"][1] < mask_array.shape[1] / 2 else "right",
    ]
    return mask_info


def _get_mask_segment_meta_2(mask_nd):
    true_indices = np.argwhere(mask_nd)

    info = {
        "area": len(true_indices),
        "area_ratio": len(true_indices) / (mask_nd.shape[0] * mask_nd.shape[1]),
        "bbox": [
            true_indices[:, 1].min(),
            true_indices[:, 0].min(),
            true_indices[:, 1].max(),
            true_indices[:, 0].max(),
        ],
        "centroid": [true_indices[:, 1].mean(), true_indices[:, 0].mean()],
    }

    top_bottom = "center"
    left_right = "center"

    if info["centroid"][1] < mask_nd.shape[0] * 0.475:
        top_bottom = "top"
    elif info["centroid"][1] > mask_nd.shape[0] * 0.525:
        top_bottom = "bottom"

    if info["centroid"][0] < mask_nd.shape[1] * 0.475:
        left_right = "left"
    elif info["centroid"][0] > mask_nd.shape[1] * 0.525:
        left_right = "right"

    info["location"] = top_bottom + "-" + left_right
    info["area_percent"] = (
        str(math.ceil((info["area_ratio"] * 100)) // 1) + "%"
    )  # round to nearest percent
    return info


def get_mask_image(mask_nd, padding=10):
    # Find the indices of all True values in nd_array using np.argwhere
    true_indices = np.argwhere(mask_nd)

    # Initialize an empty mask array with the same shape as nd_array
    mask_array = np.zeros_like(mask_nd, dtype=bool)

    # Iterate over the true indices and add padding to the mask array
    for idx in true_indices:
        # Add padding of size padding_size around the index using np.pad
        padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
        # set the padded slice to True if the original index is on the background segment
        mask_array[padded_slice] = True

    visual_mask = (mask_array * 255).astype(np.uint8)
    mask_image = Image.fromarray(visual_mask)
    return mask_image


def get_image_segments(image_path):
    processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        "facebook/maskformer-swin-base-coco"
    )

    image = Image.open(image_path)

    image = image.resize((512, 512))
    # prepare image for the model

    inputs = processor(images=image, return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # you can pass them to processor for postprocessing
    results = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    segments_info = results["segments_info"]

    segments_full = []
    instances_counter = defaultdict(int)
    for segment in segments_info:
        nd_mask_array = (
            results["segmentation"].numpy() == segment["id"]
        )  # numpy array of the mask

        mask_metadata = _get_mask_segment_meta_2(nd_mask_array)

        mask_image = get_mask_image(nd_mask_array)

        label = model.config.id2label[segment["label_id"]]
        label_name = label + "_" + str(instances_counter[label])

        mask_image_path = f"mask-of-{label_name}-{str(uuid.uuid4())[0:8]}.png"

        mask_image.save(mask_image_path)

        segments_full.append(
            {
                "label_id": segment["label_id"],
                "label_name": label_name,
                "id": segment["id"],
                "mask_path": mask_image_path,
                "mask_info": mask_metadata,
            }
        )

        instances_counter[label] += 1

    return segments_full, instances_counter


def get_image_xray_dict(image_path: str) -> dict:
    """Get image xray dict."""
    xray_dict = {}

    xray_dict["image_path"] = image_path
    xray_dict["image_contents_description"] = get_image_contents_description_1(
        image_path
    )
    xray_dict["image_contents_description_2"] = get_image_contents_description_2(
        image_path
    )
    xray_dict["image_segments"], xray_dict["instances_counter"] = get_image_segments(
        image_path
    )

    return xray_dict


def get_image_xray_summary(image_path: str) -> str:
    """Get image xray summary."""
    xray_dict = get_image_xray_dict(image_path)
    instances_counter = xray_dict["instances_counter"]

    instances_counter_str = ", ".join(
        [str(v) + " " + k for k, v in instances_counter.items()]
    )

    # create a json with label, area_percent, location, mask_path
    segments_str = ""

    for segment in xray_dict["image_segments"]:
        segment_dict = {
            "label": segment["label_name"],
            "area_percent": segment["mask_info"]["area_percent"],
            "location": segment["mask_info"]["location"],
        }
        segments_str += json.dumps(segment_dict) + "\n"

    xray_dict = get_image_xray_dict(image_path)

    summary = (
        f"The image: {xray_dict['image_path']} of '{xray_dict['image_contents_description']}' contains {instances_counter_str}. "
        f"and objects json: {segments_str}"
    )

    return summary
