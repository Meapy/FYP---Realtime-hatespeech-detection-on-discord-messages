import os
from datetime import time

import Classifier
import imageProcessing

import numpy as np
import discord

import requests
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
    for guild in guilds:
        print(f'{guild.name}(id: {guild.id})')


@client.listen()
async def on_message(message):
    channel = message.channel
    question = Classifier.process_msg(str(message.content))

    response = Classifier.predict_class([question])
    if not message.author.bot:
        if response:
            if response[0] == 0:
                await message.add_reaction(str('❌'))  # red x emoji
            elif response[0] == 1:
                await message.add_reaction(str('🔴'))  # red circle eomji

    # check images for hate speech and offensive content
    if message.attachments:
        img_data = requests.get(message.attachments[0].url).content
        with open('data/images/image_name.jpg', 'wb') as handler:
            handler.write(img_data)
        img = imageProcessing.setup_image('data\images\image_name.jpg')
        text = imageProcessing.convert_to_text(img)
        text = Classifier.process_msg(text)

        response = Classifier.predict_class([text])
        if not message.author.bot:
            if response:
                if response[0] == 0:
                    await message.add_reaction(str('❌'))  # red x emoji
                elif response[0] == 1:
                    await message.add_reaction(str('🔴'))  # red circle eomji


@client.event
async def on_raw_reaction_add(reaction):
    channel_id = reaction.channel_id
    channel = client.get_channel(channel_id)
    # if the reaction is ❗, save the message to a new file
    if not reaction.member.bot:
        if reaction.emoji.name == '❗':
            # add the message to data/reports/neither.txt if the message is not already in the file
            with open('data/Reports/neither.txt', 'r') as file:
                message = await channel.fetch_message(reaction.message_id)
                if str(message.content) not in file.read():
                    with open('data/Reports/neither.txt', 'a') as f:
                        f.write(f'{Classifier.process_msg(str(message.content))}\n')
                else:
                    print("Message already in file")
        if reaction.emoji.name == '🔴':
            # add the message to data/reports/neither.txt if the message is not already in the file
            with open('data/Reports/offensive.txt', 'r') as file:
                message = await channel.fetch_message(reaction.message_id)
                if str(message.content) not in file.read():
                    with open('data/Reports/offensive.txt', 'a') as f:
                        f.write(f'{Classifier.process_msg(str(message.content))}\n')
                else:
                    print("Message already in file")
        if reaction.emoji.name == '❌':
            # add the message to data/reports/neither.txt if the message is not already in the file
            with open('data/Reports/hatespeech.txt', 'r') as file:
                message = await channel.fetch_message(reaction.message_id)
                if str(message.content) not in file.read():
                    with open('data/Reports/hatespeech.txt', 'a') as f:
                        f.write(f'{Classifier.process_msg(str(message.content))}\n')
                else:
                    print("Message already in file")


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


kick_dict = {'username': 'counter'}
voted_dict = {'username': 'voted for'}


@client.command()
async def check(ctx, url):
    img_data = requests.get(url).content
    with open('data/images/image_name.jpg', 'wb') as handler:
        handler.write(img_data)
    img = imageProcessing.setup_image('data\images\image_name.jpg')
    text = imageProcessing.convert_to_text(img)
    await ctx.send(text)


@client.command(pass_context=True)
async def votekick(ctx, userName: discord.User):
    member = ctx.me
    voter = ctx.message.author.name
    if voter not in voted_dict:  # check if user has voted before or not, if not then add him to voted tuple
        voted_dict.update({voter: 'user'})

    if str(voted_dict[voter]) == str(userName.name):  # check if the user has voted for the same user before
        await ctx.send("You have already voted!")
    else:  # add the vote
        if str.lower(userName.name) == str.lower('Pillow'):
            await ctx.send(f'You can not vote to mute the creator')
        elif userName.name not in kick_dict:
            kick_dict.update({userName.name: 1})
            voted_dict.update({voter: userName.name})
        else:
            kick_dict[userName.name] += 1
        await ctx.send(f'{kick_dict[userName.name]}/4 people have voted to kick {userName.display_name}')

    if kick_dict[userName.name] == 4:  ##once reaches limit, kicks user
        await ctx.send(f'{userName.display_name} has been kicked')
        kick_dict.update({userName.name: 0})
        # await discord.Guild.kick(member.guild, userName)


mute_dict = {'username': 'counter'}
mvoted_dict = {'username': 'voted for'}


@client.command(pass_context=True)
async def votemute(ctx, userName: discord.Member):
    voter = ctx.message.author.name
    bot = ctx.me
    if voter not in mvoted_dict:
        mvoted_dict.update({voter: 'user'})

    if str(mvoted_dict[voter]) != str(userName.name):
        if str.lower(userName.name) == str.lower('Pillow'):
            await ctx.send(f'You can not vote to mute the creator')
        elif userName.name not in mute_dict:
            mute_dict.update({userName.name: 1})
            mvoted_dict.update({voter: userName.name})
            await ctx.send(f'{mute_dict[userName.name]}/4 people have voted to mute {userName.display_name}')
        else:
            mute_dict[userName.name] += 1
            await ctx.send(f'{mute_dict[userName.name]}/4 people have voted to mute {userName.display_name}')

    else:
        await ctx.send("You have already voted!")

    try:
        if mute_dict[userName.name] == 4:
            mute_dict.update({userName.name: 0})
            role = discord.utils.get(ctx.guild.roles, name="Muted")
            await userName.add_roles(role)
            embed = discord.Embed(title="User Muted!",
                                  description="**{0}** was muted by **{1}**!".format(userName.name, bot),
                                  color=0xff00f6)
            await ctx.send(embed=embed)
            time.sleep(60)
            await userName.remove_roles(role)
    except KeyError:
        print("person not in dict")


@has_permissions(kick_members=True)
@client.command(pass_context=True)
async def mute(ctx, userName: discord.Member):
    bot = ctx.me
    server = bot.guild
    role = discord.utils.get(ctx.guild.roles, name="Muted")
    await ctx.send("**{0}** was muted by **{1}**!".format(userName.name, bot))
    await userName.add_roles(role)


@has_permissions(kick_members=True)
@client.command(pass_context=True)
async def unmute(ctx, userName: discord.Member):
    member = ctx.me
    server = member.guild
    role = discord.utils.get(ctx.guild.roles, name="Muted")
    await ctx.send("**{0}** was unmuted".format(userName.name))
    await userName.remove_roles(role)


client.run(TOKEN)
