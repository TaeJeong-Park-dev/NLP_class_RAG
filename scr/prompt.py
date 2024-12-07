from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# Few-shot 예제
examples = [
    # {
    #     "question": "중국어권지역학전공 교양 몇 학점 들어야 돼?",
    #     "answer": "중국어권지역학전공은 교필 11학점, 교선 22학점으로 총 33학점 이수해야 돼!"
    # },
    # # 다른 예제 추가 가능
]

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
