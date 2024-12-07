import discord
from discord.ext import commands
import os
import asyncio
from config import DISCORD_BOT_TOKEN, OPENAI_API_KEY
from langchain_community.chat_models import ChatOpenAI
from retriever import retriever
from prompt import prompt
from chains import create_qa_chain
from memory import create_memory
from utils import process_qa


# dotenv_path = '/usr/workspace/.env'
# load_dotenv(dotenv_path)
# OpenAI 및 Discord 토큰 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# LLM 설정
llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Discord 봇 설정
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True

bot = commands.Bot(command_prefix='#', intents=intents)

# 사용자별 메모리와 QA 체인 저장소
user_memories = {}
user_qa_chains = {}

# 비동기 잠금 설정
lock = asyncio.Lock()

@bot.event
async def on_ready():
    print(f'Logged in as: {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    user_id = str(message.author.id)

    async with lock:
        async with message.channel.typing():
            try:
                if user_id not in user_memories:
                    user_memories[user_id] = create_memory(llm)
                    user_qa_chains[user_id] = create_qa_chain(llm, retriever, user_memories[user_id], prompt)

                response = await asyncio.to_thread(
                    process_qa,
                    user_qa_chains[user_id],
                    message.content
                )

                if response:
                    await message.channel.send(response)
                else:
                    await message.channel.send("죄송합니다. 응답을 생성하는 데 문제가 발생했습니다.")

            except Exception as e:
                print(f"Error: {e}")
                await message.channel.send("요청을 처리하는 동안 오류가 발생했습니다.")

@bot.command()
async def clear(ctx):
    user_id = str(ctx.author.id)
    if user_id in user_memories:
        del user_memories[user_id]
        del user_qa_chains[user_id]
        await ctx.send("대화 기록이 초기화되었습니다.")
    else:
        await ctx.send("초기화할 대화 기록이 없습니다.")

@bot.command()
@commands.has_role("관리자")
async def logout(ctx):
    await ctx.send("Logging out...")
    await bot.close()

bot.run(DISCORD_BOT_TOKEN)
