from langchain_old.agents.initialize import initialize_agent
from langchain_old.agents.tools import Tool
from langchain_old.chains.conversation.memory import ConversationBufferMemory
from langchain_old.llms.openai import OpenAI
from photo_gpt.tools.image_xray import get_image_xray_summary
from photo_gpt.tools.instruct_pixpix import Pix2Pix
from photo_gpt.tools.mask_former import MaskFormer
from photo_gpt.tools.stable_diffusion import StableDiffusion
from photo_gpt.tools.stable_diffusion_inpaint import StableDiffusionInpaint


class ConversationBot:
    def __init__(self, telegram_helper):
        self.telegram_helper = telegram_helper
        self.llm = OpenAI(temperature=0)

        self.masker = MaskFormer()
        self.stable_diffusion = StableDiffusion(telegram_helper)
        self.stable_diffusion_inpaint = StableDiffusionInpaint(
            self.llm, telegram_helper, self.masker
        )
        self.pix2pix = Pix2Pix(telegram_helper)
        self._inialize_langchain()

    def _inialize_langchain(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        self.tools = [
            Tool(
                name="Get photo description",
                func=get_image_xray_summary,
                description="useful for when you want to know what is inside the photo. receives image_path as input.",
            ),
            Tool(
                name="Generate image from user input text",
                func=self.stable_diffusion.run,
                description="useful for when you want to generate an image from a user input text and it saved it to a file.",
            ),
            Tool(
                name="Send a photo to the user from an image_path",
                func=self.telegram_helper.send_photo_to_user,
                description="useful for when sending the image back to user after editing or creating. "
                "receives the required image_path as input.",
            ),
            Tool(
                name="Remove something from the photo",
                func=self.stable_diffusion_inpaint.remove_part_of_image,
                description="useful for when you want to remove and object or something from the photo from its description or location. "
                "It takes a json as an input with the following schema:"
                "'image_path': '<path_to_original_image>', 'to_remove': '<item_name_or_description_or_location>'",
            ),
            Tool(
                name="Replace something from the photo",
                func=self.stable_diffusion_inpaint.replace_part_of_image,
                description="useful for when you want to replace an object from the object description or location with another object from its description."
                "It takes a json as an input with the following schema:"
                "'image_path': '<path_to_original_image>' , 'to_replace': '<item_name_or_description_or_location>', 'replace_with': '<item_name_or_description>'",
            ),
            Tool(
                name="Instruct image using text",
                func=self.pix2pix.change_style_of_image,
                description="useful for when you want to the style of the image to be like the text. like: make it look like a painting. or make it like a robot."
                "It takes a json as an input with the following schema:"
                "'image_path': '<path_to_original_image>' , 'instruct_text': '<instruct_text>'",
            ),
        ]

        self.bot_conv_agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
        )

    def get_agent(self):
        return self.bot_conv_agent
