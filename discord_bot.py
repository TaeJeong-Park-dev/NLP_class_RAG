import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# LangChain 및 OpenAI 모듈 불러오기
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.typing = False
intents.presences = False


# 벡터 저장소 및 QA 체인 설정
loader = TextLoader('/usr/workspace/plus.txt',encoding = 'utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.from_documents(docs, embeddings)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

executor = ThreadPoolExecutor(max_workers=5)

bot = commands.Bot(command_prefix='#', intents=intents)# #뒤에 명령어를 넣어서 실행시키겠다는 의미 예를 들어 #info, #hi

@bot.event
async def on_ready():# async가 붙는 거는 비동기로 실행되는 함수라는 의미. 여기서 비동기하는 것은 현재 함수가 끝나는 것을 기다리지 않고 다음 함수를 실행할 수 있도록 하는 프로그램 방식
    print(f'Logged in as: {bot.user}')# 봇이 시작될 때 실행되는 이벤트 함수다.

@bot.command()
async def hello(ctx):
    await ctx.send('Hi!')# 이 명령어를 작성하면 Hi! 메세지가 온 채널에 전송된다.
    # 채팅창에 #hello를 사용자가 보내면 실행.

# @commands.is_owner()
# @commands.has_role("관리자")
@bot.command()
async def logout(ctx):
    await ctx.send("Logging out...") # 채널에 로그아웃 메시지를 보냅니다.
    await bot.close() # 봇을 로그아웃합니다.

@bot.command()
async def ask(ctx, *, question):
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(executor, qa_chain.run, question)
        await ctx.send(response)
    except Exception as e:
        await ctx.send("요청을 처리하는 동안 오류가 발생했습니다.")
        print(f"Error: {e}")

# 봇을 생성할 때 생성된 토큰 값.
bot.run(DISCORD_BOT_TOKEN)
