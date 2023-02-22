class TelegramHelper:
    chat_id = None

    def __init__(self, bot):
        self.bot = bot

    def send_photo_to_user(self, image_path: str) -> str:
        """Run query through CLIP and parse result."""

        try:
            with open(image_path.strip(), "rb") as image:
                self.bot.send_photo(chat_id=self.chat_id, photo=image)
        except Exception as e:
            print(e)
            return "Could not send image to user. please make sure the input image_path is correct."

        return f"Image '{image_path}' has been sent to the user."

    def set_chat_id(self, chat_id):
        """Set chat_id for the model."""
        self.chat_id = chat_id
