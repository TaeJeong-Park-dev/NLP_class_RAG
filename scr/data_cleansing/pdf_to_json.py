# -- pdf 텍스트 추출 코드 --
from PyPDF2 import PdfReader
import os

pdf_path = "/usr/workspace/data/raw/gyogyng_curriculum_2024.pdf"

output_dir = "/usr/workspace/data/processed/"
output_file = os.path.join(output_dir, "output_text.txt")

# 디렉토리 생성 (없으면 생성)
os.makedirs(output_dir, exist_ok=True)

# PDF Reader 객체 생성
reader = PdfReader(pdf_path)

# 구분선 정의
separator = "-" * 50  # 50개의 "-"로 이루어진 구분선

# 텍스트 추출 및 저장
with open(output_file, "w", encoding="utf-8") as f:
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        f.write(f"Page {page_num}:\n{text}\n\n")
        f.write(f"{separator}\n")  # 페이지 구분선 추가
        

# # 모든 페이지의 텍스트 추출
# for page_num, page in enumerate(reader.pages, start=1):
#     text = page.extract_text()
#     print(f"Page {page_num}:\n{text}\n")

# # -- 추출된 텍스트 저장 --
# with open(output_file, "w", encoding="utf-8") as f:
#     for page in reader.pages:
#         text = page.extract_text()
#         f.write(text + "\n")
print("텍스트가 output_text.txt 파일에 저장되었습니다.")
