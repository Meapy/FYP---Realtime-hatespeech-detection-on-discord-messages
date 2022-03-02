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


@client.listen()
async def on_message(message):
    channel = message.channel
    question = str(message.content)

    response = Classifier.predict_class([question])
    if channel.name != 'log':
        if not message.author.bot:
            if response:
                if response[0] == 0:
                    await message.add_reaction(str('❌'))  # red x emoji
                elif response[0] == 1:
                    await message.add_reaction(str('🔴'))  # red circle eomji


@client.command(pass_context=True)
async def test(ctx, *, message):
    channel = ctx.channel

    question = Classifier.predict_class([message])  # Message gets sent to the classifier for prediction
    print(question)
    if question[0] == 0:
        await message.add_reaction(str('\❌'))
    elif question[0] == 1:
        await message.add_reaction(str('\🔴'))
    else:
        await message.add_reaction(str('\🟢'))


@client.command(help="| Clears inputted amount of messages or 5 by default")
# @has_permissions(administrator=True)
async def clear(ctx, amount=5):
    member = ctx.message.author
    if amount <= 0:
        await ctx.send(f'Enter a value greater than 0 {member.mention}, ROGER ROGER')
        print(f'{member} has entered a value less than 0 for clear')
    else:
        await ctx.channel.purge(limit=amount + 1)
        await ctx.send(f'Bot has cleared {amount} messages for {member.mention}, ROGER ROGER')
        print(f'Bot has cleared {amount} messages for {member}')

client.run(TOKEN)