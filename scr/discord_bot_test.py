import discord
from discord.ext import commands
import openai
import numpy as np
import faiss
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# (.env 파일) 로드
load_dotenv()

# OpenAI 및 Discord 토큰 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)


# ----- 문서 준비 -----
# 데이터 파일 경로
html_file_path = "/usr/workspace/raw/output.html"

# 문서 로드 및 분할
loader = UnstructuredHTMLLoader(html_file_path)
loaded_documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(loaded_documents)

# 문서와 텍스트 저장
documents = [doc.page_content for doc in split_docs]

# ----- OpenAI 임베딩 생성 -----
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# ----- FAISS 리트리버 초기화 -----
# FAISS 인덱스 생성 및 추가
vectorstore = FAISS.from_texts(documents, embedding=embeddings)
# 리트리버 생성
retriever = vectorstore.as_retriever()

# ----- Few-shot 템플릿 설정 -----
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


examples = """
예제 1:
문서: 창의적 문제 해결 역량의 주요 과목은 '상상속의 아이디어'와 '융합의 수학'입니다.
질문: 창의적 문제 해결 역량의 주요 과목이 무엇인가요?
답변: 창의적 문제 해결 역량의 주요 과목은 '상상속의 아이디어'와 '융합의 수학'입니다.

예제 2:
문서: 응용적 역량의 과목으로 '영화속의 건축여행'이 있습니다.
질문: 응용적 역량의 과목 중 하나를 알려주세요.
답변: 응용적
"""

# 예제
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\n{answer}"
)

# 프롬프트 템플릿
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""
너는 상명대학교 수강신청에 관한 질문에 답변하는 도우미야. 또한 일상적인 대화도 자연스럽게 이어나갈 수 있어.
지침:
- **무조건 한국어로** 친숙한 어투로 반말로 답변해.
""",
    suffix="""
대화 기록: {chat_history}
질문: {question}
컨텍스트: {context}

답변:
""",
    input_variables=["question", "context", "chat_history"]
)

# ----- 메모리 및 체인 생성 함수 -----
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain

def create_memory(llm):
    return ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000,
        output_key="answer"
    )

def create_qa_chain(llm, retriever, memory, prompt):
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

# # ----- 5. 문서 검색 및 답변 생성 -----
# def get_rag_answer(query):
#     """
#     리트리버와 LangChain FewShotPromptTemplate을 통합하여 답변 생성
#     """
#     try:
#         # 리트리버로 관련 문서 검색
#         related_docs = retriever.get_relevant_documents(query)
#         context = "\n\n".join([doc.page_content for doc in related_docs])

#         # Few-shot 템플릿을 사용하여 프롬프트 생성
#         prompt = fewshot_template.format(context=context, question=query)

#         # ChatOpenAI 호출
#         response = chat([HumanMessage(content=prompt)])
#         return response.content.strip()
#     except Exception as e:
#         return f"오류 발생: {e}"

# ----- 디스코드 봇 초기화 -----
# Discord 봇 설정
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True
# 사용자별 메모리와 QA 체인 저장소
user_memories = {}
user_qa_chains = {}

bot = commands.Bot(command_prefix='#', intents=intents)

# 비동기 잠금 설정(무한루프 때메)
lock = asyncio.Lock()
bot = commands.Bot(command_prefix='#', intents=intents)

# ----- 디스코드 봇 이벤트 -----
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
@commands.has_role("관리자")
async def logout(ctx):
    await ctx.send("Logging out...")
    await bot.close()

# ----- 7. 봇 실행 -----
bot.run(DISCORD_BOT_TOKEN)
