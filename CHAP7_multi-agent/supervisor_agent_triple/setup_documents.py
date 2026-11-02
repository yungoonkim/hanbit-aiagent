# ============================================================================
# [7.5 핸드오프 도구 기반 3중 슈퍼바이저] 사내 문서 벡터 DB 구축 (setup_documents.py)
# 비유: 사내 도서관(Chroma DB)을 개관하기 전에 PDF와 워드(.docx) 문서들을 적당한 크기로 잘라서
#       파일명 꼬리표(메타데이터)를 붙여 서가에 꽂아두는 사전 준비 스크립트입니다.
# ============================================================================
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    # [ 1 ] 원본 문서가 들어 있는 폴더 경로와 벡터 DB를 저장할 폴더 경로를 지정합니다.
    DOCUMENTS_DIR = "./supervisor_agent_triple/documents" # [ 1 ]
    DB_PATH = "./supervisor_agent_triple/chroma_db"

    document_paths = []
    if os.path.exists(DOCUMENTS_DIR):
        for file_name in os.listdir(DOCUMENTS_DIR):
            file_path = os.path.join(DOCUMENTS_DIR, file_name)
            if os.path.isfile(file_path):
                document_paths.append(file_path)

    all_docs = []

    # [ 2 ] 각 파일의 확장자(.pdf, .docx/.doc)에 맞는 문서 로더로 텍스트를 읽어옵니다.
    for file_path in document_paths: # [ 2]
        if os.path.exists(file_path):
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                pages = []
                for page in loader.lazy_load():
                    pages.append(page)

            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
                pages = loader.load()

            # [ 3 ] 긴 문서를 1,000자 단위(200자 중첩)로 쪼개어 검색 정확도를 높입니다.
            text_splitter = RecursiveCharacterTextSplitter( # [ 3 ]
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(pages)

            file_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(file_name)[0]

            # [ 4 ] 각 문서 조각(청크) 본문 앞에 문서명을 명시하고, 메타데이터에도 파일 정보를 꼼꼼히 기록합니다.
            for doc in docs: # [ 4 ]
                original_content = doc.page_content
                doc.page_content = f"[문서명: {file_name}]\n\n{original_content}"

                doc.metadata.update({
                    'file_name': file_name,
                    'file_name_without_ext': file_name_without_ext,
                    'file_extension': file_extension,
                    'file_type': 'pdf' if file_extension == '.pdf' else 'word'
                })

            all_docs.extend(docs)

    # [ 5 ] 모든 문서 청크를 임베딩하여 Chroma 벡터 데이터베이스에 영구 저장합니다.
    vectorstore = Chroma.from_documents( # [ 5 ]
        documents=all_docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=DB_PATH,
        collection_name="documents"
    )

if __name__ == "__main__":
    main()
