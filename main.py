import os
import Classifier
import numpy as np
from discord import opus
from discord.ext.commands import Bot, has_permissions, CheckFailure, MissingPermissions
from discord.ext import commands
from threading import Timer
from dotenv import load_dotenv
from collections import namedtuple

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

client = commands.Bot(command_prefix='~')


@client.event
async def on_ready():
    guilds = list(client.guilds)
    print(f'{client.user} is connected to the following guilds:\n')
    print(f'{guilds[1].name}(id: {guilds[1].id})')
    # for guild in guilds:
    #     print(f'{guild.name}(id: {guild.id})')

@client.command(pass_context=True)
async def test(ctx, *, message):
    channel = ctx.channel

    question = Classifier.predict_class([message]) # Message gets sent to the classifier for prediction
    print(question)
    if question[0] == 0:
        response = "Message above has been deleted for containing hatespeech"
    elif question[0] == 1:
        response = "Message above has been deleted for containing offensive language"
    else:
        response = "No action taken"
    if not ctx.author.bot:
        await channel.send(f'{response}')

client.run(TOKEN)
