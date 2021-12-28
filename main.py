import math
import os
import io
import random
import csv
import discord
import datetime
import classifier
from discord import opus
from discord.ext.commands import Bot, has_permissions, CheckFailure, MissingPermissions
from discord.ext import commands
from threading import Timer
import time
from dotenv import load_dotenv
from collections import namedtuple

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

client = commands.Bot(command_prefix='#')


@client.event
async def on_ready():
    guilds = list(client.guilds)
    print(f'{client.user} is connected to the following guilds:\n')
    for guild in guilds:
        print(f'{guild.name}(id: {guild.id})')


# create a function that listens on every message, and then responds to it
@client.listen()
async def on_message(message):
    channel = message.channel

    question = classifier.predictions(message.content)

    if not message.author.bot:
        await channel.send(f'{question}')


client.run(TOKEN)
