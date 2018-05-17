# Chatbot-coding-assistant

This Chatbot is machine-learning based conversational dialog engine built in Python which helps you find answers your programing questions by referring to the appropriate post in Stack Overflow.


## Installation

### Install ChatterBot

This package can be installed from [PyPi](https://pypi.python.org/pypi/ChatterBot) by running:
```
python install chatterbot
```
### Setup Telegram messenger

Telegram messagenger will be used to interact with the Chatbot. You need to obtain a token from https://telegram.org/ by:
* Searching for @BotFather
* Using "/newbot" to create a new bot

## Basic Usage

```
python coding_bot.py --token <insert your telegram token
```

# Training data
The bot uses two sets of training data:
* One for normal converstation (eg: Hi ..)
* One for technical conversation to find stackoverflow posts for programing questions    

## REFERENCES
ChatterBot https://github.com/gunthercox/ChatterBot
