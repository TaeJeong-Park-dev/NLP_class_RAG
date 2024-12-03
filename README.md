# NLP_class_RAG

요구 사항
- Python 3.8 이상
- Discord 계정 및 봇 생성
- OpenAI API Key
- MacOS 환경

주요 기능
- 졸업 요건 질문 응답: 사용자 질문에 맞는 졸업 요건 정보를 제공
- 대화 기록 관리: 이전 대화를 기억하고 자연스러운 응답 생성
- 디스코드 명령어
  #clear: 대화 기록 초기화
  #logout: 봇 종료 (관리자 전용)

주요 코드 구조
문서 로더: PDF 파일에서 데이터를 로드
벡터 저장소: FAISS를 사용하여 문서를 벡터화 및 검색 가능하게 만듬
LangChain: OpenAI GPT 모델을 기반으로 대화형 체인을 생성
Discord Integration: Discord.py를 사용하여 디스코드와 통신

설치
- Python 설치
- 필요한 라이브러리
  ```
  pip install python-dotenv discord.py langchain openai faiss-cpu
  ```
- .env 파일 내용
  ```
  DISCORD_BOT_TOKEN=your_discord_bot_token
  OPENAI_API_KEY=your_openai_api_key
  ```
- 사용할 PDF 파일 프로젝트의 /usr/workspace/ 디렉토리에 저장
- Discord 봇 설정
  ```
  https://discord.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=517543&scope=bot
  ```
- 프로젝트 실행
  ```
  /usr/local/bin/python /usr/workspace/discord_bot_faiss.py
  ```
