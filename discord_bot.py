import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True

# PDF 파일 로드 및 처리
# pdf_path = '/usr/workspace/gyogyng_curriculum_2024.pdf'
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large") 

vectorstore = Chroma(
    persist_directory="/usr/workspace/sample_2018",
    embedding_function=embeddings,
    collection_name="excel_embedding"  # ChromaDB에서 생성한 collection 이름과 동일하게 설정
)

retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    """
    You are an assistant specialized in answering questions about 상명대학교-related Course registration, but you also engage in casual daily conversations naturally.

    Guidelines:
    - You have the knowledge of a general adult and natural conversational ability in Korean.
    - If the question is school-related and not found in the provided context, reply with:
      "죄송하지만 해당 내용은 제가 알 수 있는 정보에 포함되어 있지 않습니다."
    - Always remember previous conversations when answering questions.

    Chat History: {chat_history}
    Question: {question}
    Context: {context}

    Answer:
    """
)


llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o")

user_memories = {}
user_qa_chains = {}

def create_qa_chain(memory):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        return_generated_question=False,
        verbose=True
    )

def process_qa(chain, question):
    try:
        result = chain({"question": question})
        return result.get("answer", "응답을 생성할 수 없습니다.")
    except Exception as e:
        print(f"QA Chain Error: {e}")
        return None

lock = asyncio.Lock()
bot = commands.Bot(command_prefix='#', intents=intents)

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
                    user_memories[user_id] = ConversationSummaryBufferMemory(
                        llm=llm,
                        memory_key="chat_history",
                        return_messages=True,
                        max_token_limit=2000,
                        output_key="answer"
                    )
                    user_qa_chains[user_id] = create_qa_chain(user_memories[user_id])

                response = await asyncio.to_thread(
                    process_qa,
                    user_qa_chains[user_id],
                    message.content
                )
                
                if response:
                    # 메모리에 대화 저장
                    await message.channel.send(response)
                else:
                    await message.channel.send("죄송합니다. 응답을 생성하는 데 문제가 발생했습니다.")

            except Exception as e:
                print(f"Error: {e}")
                await message.channel.send("요청을 처리하는 동안 오류가 발생했습니다.")

@bot.command()
async def clear(ctx): #clear_history에서 이름을 직관적이게 바꿔봤다
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