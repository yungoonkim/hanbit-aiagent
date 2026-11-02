# ============================================================================
# [6.5절] 벡터 데이터베이스(Chroma) 기반 문서 검색기 (rag_agent/retriever.py)
# ============================================================================
# 이 파일은 미리 '한글 맞춤법 및 표준어 규정 해설' PDF 문서를 숫자 벡터로 변환해 저장해 둔
# 크로마(Chroma) 데이터베이스를 불러와서, 에이전트가 쓸 수 있는 '검색 도구(retriever_tool)'로 만듭니다.
#
# 💡 비전공자를 위한 일상 비유:
# - **벡터 데이터베이스(Chroma)**: 책의 모든 문장을 '의미 좌표(숫자 배열)'로 바꿔 정리해 둔 **초고속 디지털 도서관**입니다.
# - **리트리버(Retriever)**: 질문과 가장 의미가 비슷한 문단 3개(`k=3`)를 순식간에 뽑아오는 **도서관 사서**입니다.
# - **create_retriever_tool**: 도서관 사서에게 호출벨(`pdf_search`)을 달아주어 AI가 필요할 때 부를 수 있게 만든 것입니다.
# ============================================================================

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

from dotenv import load_dotenv

# .env 파일에서 API 키를 불러옵니다.
load_dotenv()


# [ 1 ] 미리 PDF 문서를 쪼개어 저장해 둔 로컬 벡터 DB 폴더 경로입니다.
DB_PATH = "./rag_agent/chroma_db" # [ 1 ]

# 저장된 크로마(Chroma) 벡터 저장소를 불러옵니다.
# 주의: 문서를 저장할 때 사용한 임베딩 모델(text-embedding-3-small)과 동일한 모델을 써야 좌표 체계가 맞습니다!
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="korean_pdf",
)

# 벡터 저장소가 정상적으로 연결되었는지 데이터를 조회해 봅니다.
vectorstore.get()

# [ 2 ] 벡터 저장소를 '검색기(Retriever)'로 변환합니다.
# search_kwargs={"k": 3}은 "질문과 가장 관련성이 높은 문서 조각 상위 3개를 가져와라!"라는 뜻입니다.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # [ 2 ]

# [ 3 ] AI(챗봇 노드)가 도구 호출(Tool Call) 방식으로 사서를 부를 수 있도록 도구(Tool)로 포장합니다.
retriever_tool = create_retriever_tool( # [ 3 ]
    retriever,
    name="pdf_search",
    description="use this tool to search information from the Korean Spelling Rules PDF document",
)

if __name__ == "__main__":
    # 단독 실행 테스트: '부엌' 발음 규정을 PDF에서 잘 찾아오는지 확인합니다.
    response = retriever_tool.invoke("부엌 이 들어간 경우 어떻게 발음하나요?")
    print(response)
