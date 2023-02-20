# import json
#
# from PIL import Image
#
# from langchain import OpenAI
# from photo_gpt.tools.telegram_utls import TelegramHelper
#
# llm = OpenAI(temperature=0)
# masker = MaskFormer()
#
#
# telegram_helper = TelegramHelper()
#
#
# stable_diffusion_inpaint = StableDiffusionInpaint(
#     IMAGE_STORE, llm, telegram_helper, masker
# )
# stable_diffusion = StableDiffusion(IMAGE_STORE, telegram_helper)
#
# stable_diffusion.run("cute cat playing in the park")
# pix2pix = Pix2Pix(IMAGE_STORE, telegram_helper)
#
# remove_from_image_json = {
#     "image_path": "2023-02-19 17.50.47.jpg",
#     "to_remove": "marked area",
# }
#
# # stable_diffusion_inpaint.remove_part_of_image(json.dumps(remove_from_image_json))
#
# replace_in_image_json = {
#     "image_path": "88855.jpg",
#     "to_replace": "area painted in red",
#     "replace_with": "a cat playing with a ball",
# }
#
#
# # stable_diffusion_inpaint.replace_part_of_image(json.dumps(replace_in_image_json))
# mask = Image.open("fallback_mask.png")
# original_image = Image.open("88855.jpg")
# # result = stable_diffusion_inpaint.inpaint(original_image, mask, "snow man")
#
# # result.save("result.png")
#
#
# pix2pix_json = {
#     "image_path": "63ad6740.png",
#     "instruct_text": "make it in the style of oil painting",
# }
# # pix2pix.change_style_of_image(json.dumps(pix2pix_json))
#
# # stable_diffusion_inpaint.replace_part_of_image(json.dumps(replace_in_image_json))
#
#
# # print(get_image_xray_summary("created_image_d723c48c.png"))
#
# # stable_diffusion.run("a dog in the park")
#
# # find stuff in image
# # image_1 = Image.open("f3af6bf8.png")
# # image_1 = image_1.convert("RGB")
# # image_1 = image_1.resize((512, 512))
# # mask_former = MaskFormer(image_store={})
# # image = mask_former.get_mask_from_text("f3af6bf8.png", "cat")
# # result = stable_diffusion_inpaint.inpaint(image_1, mask, "red cat")
# # result.save("result.png")
