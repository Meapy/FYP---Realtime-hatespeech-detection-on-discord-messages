## Using machine learning to detect hate speech and offensive language through instant messaging
My project is going to be a machine learning model that when trained using appropriate data will be able to detect hate speech and offensive messages. To test and showcase the possibilities of what to do once the messages have been detected, I will be using the Discord API to make a discord bot. I will be using the discord.py wrapper which is written in python.

## Motivation
Since the invention of the internet, one of the most used technologies is instant messaging. While this is not bad in anyway, it brings along a new way to spread offensive and hateful messages and statements. Especially in a medium such as Discord where there are many young users and no built-in moderation or detection of such messages. Users can be exposed to very hateful and offensive messages at an age where they should not. This is the problem I am attempting to solve in this final year project. I am going to be using Machine Learning and NLP to detect Hate speech and Offensive messages real time and use the built-in tools in discord to deal with the messages through various methods. 

## Screenshots
![](https://media.discordapp.net/attachments/521374756864524289/961891264935100416/unknown.png)
![](https://media.discordapp.net/attachments/521374756864524289/961892010103558175/unknown.png)

## Tech/framework used
Tensorflow
BERT language model
Discord.py
NumPy
OpenCV
Pytesseract


<b>Built with</b>
-Python

## Features
Discord API	
Model Development	
Moderation tools	
Image to text extraction and classification	
Data collection and reporting false flags	
Multiple Language support


## Code Example
main.py
```python
import discord
from discord import opus
from discord.ext.commands import Bot
from discord.ext import commands
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

client = commands.Bot(command_prefix='~')
@client.event
async def on_ready():
    guilds = list(client.guilds)
    print(f'{client.user} is connected to the following guilds:\n')
    for guild in guilds:
        print(f'{guild.name}(id: {guild.id})')

client.run(TOKEN)
```
Model embeddings
```python
import tensorflow_hub as hub
import tensorflow_text as text
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_embeddings(sentences):
  '''return BERT-like embeddings of input text
  Args:
    - sentences: list of strings
  Output:
    - BERT-like embeddings: tf.Tensor of shape=(len(sentences), 768)
  '''
  preprocessed_text = preprocessor(sentences)
  return encoder(preprocessed_text)['pooled_output']

get_embeddings([
    "Hello, what are you doing"]
)
```
Predictions
```python
@client.listen()
async def on_message(message):
    question = Classifier.process_msg(str(message.content))

    if not message.author.bot:
        response = Classifier.predict_class([question])
        if response:
            if response[0] == 0:
                await message.add_reaction(str('❌'))  # red x emoji
            elif response[0] == 1:
                await message.add_reaction(str('🔴'))  # red circle eomji


```
```python
    if model.predict(message)[0][0] > 0.5:
        return [np.argmax(pred) for pred in model.predict(message)]
    elif model.predict(message)[0][1] > 0.85:
        return [np.argmax(pred) for pred in model.predict(message)]
```



## Installation & How to use
clone the repository
install all the required libaries, either download the pretrained model or run main.ipynb in prototype3 folder to train your own model
create a .env file with DISCORD_TOKEN={your token}
once model is trained run main.py

## API Reference




