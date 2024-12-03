from dotenv import load_dotenv
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate

from langchain.prompts.few_shot import FewShotPromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
import discord
from discord.ext import commands
import asyncio

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True

bot = commands.Bot(command_prefix='#', intents=intents)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = '/usr/workspace/졸업기준학점_2018.pdf'
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate


examples = [
    {
        "question": "중국어권지역학전공 교양 몇 학점 들어야 돼?",
        "answer": """
중국어권지역학전공은에서는 교필 11학점, 교선 22학점으로 총 33학점 이수해야 돼!
"""
    },
    {
        "question": "중국어권지역학전공은 전공 몇 학점 들어야 돼?",
        "answer": """
중국어권지역학전공은 학기 12학점, 전필 0학점, 전심 15학점, 전선 33학점으로 총 60학점 이수해야 돼!
"""
    },
    {
        "question": "전자공학과 전공 몇 학점 들어야 돼?",
        "answer": """
전자공학과는 학기 0학점, 전필 0학점, 전심 15학점, 전선 54학점으로 총 74학점 이수해야 돼!
"""
    }
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\n{answer}"
)

# Combined Prompt
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""
너는 상명대학교 수강신청에 관한 질문에 답변하는 도우미야. 또한 일상적인 대화도 자연스럽게 이어나갈 수 있어.

**컨텍스트 정보**:
- 이 컨텍스트는 상명대학교의 학과별로 졸업요건에 대한 내용을 포함하고 있어.

지침:
- 일반 성인의 지식을 가지고 있고, **무조건 한국어로** 자연스럽게 대화해.
- **친숙한 어투로 반말로** 답변해.
- 학교 관련 질문을 하면 무조건 컨텍스트 기반으로 답변해. 거기서 답변 못 하겠으면 모르겠다고 사과하고.
- 항상 이전 대화를 기억하고 질문에 답변해.

""",
    suffix="""
대화 기록: {chat_history}
질문: {question}
컨텍스트: {context}

답변:
""",
    input_variables=["question", "context", "chat_history"]
)


# Language model setup
llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

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
