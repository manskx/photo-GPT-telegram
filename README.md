
## This is just a demo of using llm (GPT) with stable diffusion and huggingface.
## If you like this project, please give it a star ⭐️ to continue the development.


# photo-GPT-telegram
is a telegram bot based on large language models (GPT-3) for image tasks like creating image, captioning, editing and 
general conversation like chatting about images based on stable diffusion and huggingface and langchain.

## How to use (tested with macos only)

1- create a telegram bot using @BotFather and get the token in env variable `TELEGRAM_BOT_TOKEN`

2- Get OpenAI API key and put it in env variable `OPENAI_API_KEY` or Cohere API key and put it in env variable `COHERE_API_KEY`

3- ```bash
   git clone https://github.com/manskx/photo-GPT-telegram.git
   cd photo-GPT-telegram
   python -m venv venv
   source venv/bin/activate
   python -m pip install -r requirements.txt```
   
4- ```bash
    export PYTHONPATH=$PWD
    cd photo_gpt
    python main.py```

5- send `/start` to your bot in telegram

# example

https://user-images.githubusercontent.com/7218339/220134549-1bb94ab8-646e-477d-919d-617b9cb78cb5.mp4


# authors
- @manskx and copilot
