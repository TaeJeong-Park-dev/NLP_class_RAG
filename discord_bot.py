import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# LangChain 및 OpenAI 모듈 불러오기
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True

# 벡터 저장소 및 QA 체인 설정
loader = TextLoader('/usr/workspace/plus.txt', encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(docs, embeddings)

llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# 대화 메모리 설정
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

# ConversationalRetrievalChain 사용하여 대화형 QA 체인 생성
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

executor = ThreadPoolExecutor(max_workers=5)
lock = asyncio.Lock()  # 메시지 잠금 객체 생성

bot = commands.Bot(command_prefix='#', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as: {bot.user}')

@bot.event
async def on_message(message):
    # 봇이 자기 자신에게 응답하지 않도록 설정
    if message.author == bot.user:
        return

    # 잠금을 사용하여 메시지 처리를 순차적으로 실행
    async with lock:
        async with message.channel.typing():  # 타이핑 효과 시작
            loop = asyncio.get_event_loop()
            try:
                # qa_chain을 사용해 질문에 대한 응답 생성
                response = await loop.run_in_executor(
                    executor, qa_chain.invoke, {"question": message.content, "chat_history": []}
                )
                await message.channel.send(response["answer"])  # 응답에서 'answer' 키를 사용하여 출력
            except Exception as e:
                await message.channel.send("요청을 처리하는 동안 오류가 발생했습니다.")
                print(f"Error: {e}")

@bot.command()
@commands.has_role("관리자")
async def logout(ctx):
    await ctx.send("Logging out...")
    await bot.close()

bot.run(DISCORD_BOT_TOKEN)
