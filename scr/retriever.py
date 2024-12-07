from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
from config import DISCORD_BOT_TOKEN, OPENAI_API_KEY

# JSON파일로 실행 코드
from langchain_community.document_loaders import JSONLoader
json_files = [
    '/usr/workspace/data/processed/졸업기준학점_2018.json'
]
all_documents = []

# JSON 파일을 반복적으로 로드
for file_path in json_files:
    year = file_path.split('_')[-1].split('.')[0]

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".",  # 전체 데이터를 로드
        text_content=False  # 데이터를 원래 형태로 유지
    )
    documents = loader.load()

    # for doc in documents: # doc은 json안에 하나의 덩어리 -> csv에서 한 줄.(예: 소프트웨어학과 일반편입 ---)
    #     doc.metadata["year"] = year  # 연도를 메타데이터로 추가
    #     all_documents.append(doc)


# 텍스트 청킹
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 임베딩과 벡터 스토어 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.from_documents(docs, embeddings)

# 검색기 생성
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 50}
)
