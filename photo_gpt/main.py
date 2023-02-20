import logging
import uuid

import telegram
from telegram import Update
from telegram.ext import Filters, MessageHandler, CommandHandler, Updater

from photo_gpt.config import TELEGRAM_BOT_TOKEN
from photo_gpt.langchain_agent import ConversationBot
from photo_gpt.tools.telegram_utls import TelegramHelper

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

telegram_helper = TelegramHelper(bot=bot)
conv_bot = ConversationBot(telegram_helper)


def handle_msg(update: Update):
    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        file = bot.get_file(file_id)
        received_image = f"{str(uuid.uuid4())[0:8]}.png"
        file.download(received_image)
        caption = update.message.caption
        user_msg = f"{caption}. image_path: {received_image}"
    else:
        user_msg = update.message.text

    # TODO: don't do this, maybe to send chat_id to langchain agent
    telegram_helper.set_chat_id(update.effective_chat.id)
    chain_response = conv_bot.get_agent().run(user_msg)
    return chain_response


def start(update: Update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a photo GPT bot, I do everything with photos!",
    )


def msg_handler(update: Update, context):
    msg = handle_msg(update)

    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


if __name__ == "__main__":
    updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler("start", start)
    echo_handler = MessageHandler((~Filters.command), msg_handler)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(echo_handler)
    updater.start_polling()
    updater.idle()
