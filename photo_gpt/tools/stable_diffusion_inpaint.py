import json
import uuid

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from photo_gpt.tools.image_xray import get_image_xray_dict
from photo_gpt.tools.mask_former import MaskFormer
from photo_gpt.tools.telegram_utls import TelegramHelper


class StableDiffusionInpaint:
    def __init__(
        self,
        llm,
        telegram_helper: TelegramHelper,
        mask_former: MaskFormer = None,
        image_store: dict = None,
    ):
        self.image_store = image_store
        self.mask_former = mask_former if mask_former else MaskFormer(image_store)
        self.llm = llm
        self.telegram_helper = telegram_helper

    def run(self, input_json) -> str:
        parsed_json = json.loads(input_json)
        original_image_path = parsed_json["original_image_path"]
        mask_image_path = parsed_json["mask_image_path"]
        text_to_replace_masked_part_with = parsed_json[
            "text_to_replace_masked_part_with"
        ]

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
        )

        pipe = pipe.to("mps")

        # Recommended if your computer has < 64 GB of RAM
        pipe.enable_attention_slicing()

        original_image = Image.open(original_image_path)
        mask_image = Image.open(mask_image_path)

        image = pipe(
            prompt=text_to_replace_masked_part_with,
            image=original_image,
            mask_image=mask_image,
        ).images[0]

        image_path = "modified_image_" + str(uuid.uuid4())[0:8] + ".png"

        image.save(image_path)
        print(
            f"Image original_image_path with mask {mask_image_path} has been modified with {text_to_replace_masked_part_with} "
            f"and saved to path: {image_path} but NOT sent to user"
        )
        return (
            f"Image original_image_path with mask {mask_image_path} has been modified with {text_to_replace_masked_part_with} "
            f"and saved to path: {image_path} but NOT sent to user"
        )

    def inpaint(self, original_image, mask_image, text_to_replace_masked_part_with):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
        )

        pipe = pipe.to("mps")

        # Recommended if your computer has < 64 GB of RAM
        pipe.enable_attention_slicing()

        image = pipe(
            prompt=text_to_replace_masked_part_with,
            image=original_image,
            mask_image=mask_image,
        ).images[0]

        return image

    def remove_part_of_image(self, input_json):
        parsed_json = json.loads(input_json.strip().replace("'", '"'))
        image_path = parsed_json["image_path"]
        to_be_removed_txt = parsed_json["to_remove"]

        original_image = Image.open(image_path)
        original_image = original_image.resize((512, 512))
        mask_image = self.find_stuff_in_image(image_path, to_be_removed_txt)
        if not mask_image:
            return f"Error: Could not find '{to_be_removed_txt}' in image {image_path}, please draw on the area that you want to remove in the image and try again."

        elif mask_image == "found_multiple":
            return f"Error: Found multiple instances of '{to_be_removed_txt}' in image {image_path}, please specify which one to remove and try again."

        updated_image = self.inpaint(original_image, mask_image, "background")

        # save the updated image
        updated_image_path = "{}_updated_{}.png".format(
            image_path.split(".")[0], str(uuid.uuid4())[0:4]
        )
        updated_image.save(updated_image_path)

        output = f"'{to_be_removed_txt}' has been removed from image {image_path} and saved to path: {updated_image_path}\n"
        output += self.telegram_helper.send_photo_to_user(updated_image_path)

        print(output)
        return output

    def replace_part_of_image(self, input_json):
        parsed_json = json.loads(input_json.strip().replace("'", '"'))

        image_path = parsed_json["image_path"]
        to_be_replaced_txt = parsed_json["to_replace"]
        replace_with_txt = parsed_json["replace_with"]

        original_image = Image.open(image_path)
        original_image = original_image.resize((512, 512))
        mask_image = self.find_stuff_in_image(image_path, to_be_replaced_txt)
        if not mask_image:
            return f"Error: Could not find '{to_be_replaced_txt}' in image {image_path}, please draw on the area that you want to replace and try again."
        elif mask_image == "found_multiple":
            return f"Error: Found multiple instances of '{to_be_replaced_txt}' in image {image_path}, please specify which one to replace and try again."
        updated_image = self.inpaint(original_image, mask_image, replace_with_txt)

        # save the updated image
        updated_image_path = "{}_updated_{}.png".format(
            image_path.split(".")[0], str(uuid.uuid4())[0:4]
        )
        updated_image.save(updated_image_path)

        output = f"'{to_be_replaced_txt}' has been replaced with '{replace_with_txt}' in image {image_path} and saved to path: {updated_image_path}\n"
        output += self.telegram_helper.send_photo_to_user(updated_image_path)

        print(output)
        return output

    def find_stuff_in_image(self, image_path, stuff_description):
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
                "mask_path": segment["mask_path"],
            }
            segments_str += json.dumps(segment_dict) + " \n"

        PROMPT = (
            f"An image of '{xray_dict['image_contents_description']}' contains {instances_counter_str}. "
            f"and objects json: {segments_str}"
        )

        PROMPT += (
            f" Let's pretend that you are a python function that searches for an object that matches "
            f"the input query: '{stuff_description}'."
            f" The function returns only the result either of \n "
            f" mask_path: if you are sure there's only one object that matches the input query. \n"
            f" not_found: if there is no object that matches the input query. \n"
            f" found_multiple: if there are more than one object that EXACTLY matches the input query. \n\n"
            f"NOTE: THE FUNCTION RETURNS THE RESULT ONLY WITHOUT EXPLANATION. Your function return is: "
        )

        llm_output = self.llm(PROMPT)
        print(f"prompt: {PROMPT}")
        print(f"llm_output: {llm_output}")

        if llm_output.strip() == "found_multiple":
            print(
                f"Found multiple objects that match {stuff_description} in image {image_path}"
            )
            return "found_multiple"

        # try to read the mask from the output
        try:
            mask_path = (
                llm_output.strip()
                .replace("mask_path: ", "")
                .replace('"', "")
                .replace("'", "")
            )
            mask = Image.open(mask_path)
            return mask
        except:
            print(f"Could not read mask from {llm_output}")
            pass

        fallback_mask = self.mask_former.get_mask_from_text(
            image_path, stuff_description
        )

        if not fallback_mask:
            print(f"Could not find '{stuff_description}' in image {image_path}")
            return None
        fallback_mask.save("fallback_mask.png")
        return fallback_mask
