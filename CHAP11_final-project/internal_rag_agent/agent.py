# ==============================================================================
# 📘 [CHAP11] 사내 지식 RAG 에이전트 본체 (InternalRAGAgent)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - 이 에이전트는 사내 도서관(Supabase 벡터/관계형 DB)을 관리하는 '수석 사서'입니다!
# - LangGraph 상태 그래프(StateGraph)로 설계되어 있으며, 두 가지 큰 갈림길을 가집니다:
#   1) intent_router(의도 파악 안내데스크):
#      - "휴가 규정 알려줘" -> `search_router`(책 찾기)로 이동
#      - "이 파일 인덱싱해줘 (gdrive://file/xxx)" -> `index_document_node`(신간 도서 등록)로 이동
#   2) search_router(검색 방식 결정):
#      - 문서 내용 질문("AI 트렌드가 뭐야?") -> `vector_search`(의미 기반 유사도 검색)
#      - 문서 속성 질문("PDF 파일 목록 보여줘") -> `sql_search`(메타데이터 조건 필터 검색)
# ==============================================================================

import os
import io
import json
import logging
from typing import TypedDict, Literal, List, Dict, Any, AsyncIterator, Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from supabase import create_client, Client

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader


logger = logging.getLogger(__name__)

load_dotenv()


INTENT_SEARCH = "search"    # 문서 검색 (기본)
INTENT_INDEX = "index"      # 신규 문서 인덱싱


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None


def get_supabase() -> Client:
    """Supabase 데이터베이스 클라이언트 싱글톤 반환"""
    global supabase
    if supabase is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL, SUPABASE_KEY 환경변수 필요")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase


# 의도 분류 및 답변 생성에 사용할 Gemini 모델 (gemini-3.8-flash)
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def get_embedding(text: str) -> List[float]:
    """텍스트를 숫자 벡터(임베딩)로 변환합니다."""
    return embeddings.embed_query(text)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """PDF 바이너리 데이터에서 페이지별로 텍스트를 추출합니다."""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []

        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[페이지 {page_num}]\n{page_text}")

        full_text = "\n\n".join(text_parts)
        logger.info(f"[RAG AGENT] PDF 텍스트 추출 완료: {len(pdf_reader.pages)}페이지, {len(full_text)}자")
        return full_text

    except Exception as e:
        logger.error(f"[RAG AGENT] [ERROR] PDF 텍스트 추출 실패: {e}")
        raise


def extract_text_from_bytes(file_bytes: bytes, mime_type: str = "") -> str:
    """
    파일 바이너리 데이터에서 MIME 타입(PDF 또는 일반 텍스트)에 맞춰 글자를 추출합니다.

    Args:
        file_bytes: 파일 바이너리 데이터
        mime_type: 파일의 MIME 타입 (예: application/pdf)

    Returns:
        추출된 텍스트 문자열
    """
    try:
        if "pdf" in mime_type.lower():
            return extract_text_from_pdf(file_bytes)
        else:
            try:
                return file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return file_bytes.decode('cp949')
                except UnicodeDecodeError:
                    return file_bytes.decode('latin-1')
    except Exception as e:
        logger.error(f"[RAG AGENT] [ERROR] 텍스트 추출 실패: {e}")
        raise


# ============================================================================
# Google Drive 읽기 전용 헬퍼 (storage_ref 기반)
# ============================================================================
# 💡 왜 RAG 에이전트 안에 읽기 전용 클라이언트가 따로 있을까요?
# File Management Agent가 알려준 파일 주소(`gdrive://file/xxx`) 열쇠만 받아서,
# 무거운 파일 데이터는 네트워크 A2A 메시지로 거치지 않고 RAG 에이전트가 직접 다운로드하기 위해서입니다!

# Google Drive API
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# 프로젝트 루트 기준 경로 (file_management_agent와 인증 파일 공유)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CREDENTIALS_PATH = os.path.join(_PROJECT_ROOT, 'file_management_agent', 'credentials.json')
_TOKEN_PATH = os.path.join(_PROJECT_ROOT, 'file_management_agent', 'token.json')


class _GDriveReadOnlyClient:
    """
    Google Drive 읽기 전용 클라이언트

    File Agent가 반환한 storage_ref (file_id)로만 접근 가능하며,
    목록 조회, 검색, 생성, 삭제 등은 수행하지 않고 오직 인덱싱용 다운로드만 담당합니다.
    """

    def __init__(self):
        self.service = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        creds = None
        if os.path.exists(_TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(_TOKEN_PATH, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(_CREDENTIALS_PATH):
                    raise FileNotFoundError(f"credentials.json not found: {_CREDENTIALS_PATH}")
                flow = InstalledAppFlow.from_client_secrets_file(_CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(_TOKEN_PATH, 'w') as token:
                token.write(creds.to_json())

        self.service = build('drive', 'v3', credentials=creds)
        self._initialized = True
        logger.info("[RAG AGENT] [INIT] Google Drive 읽기 전용 클라이언트 초기화 완료")

    def download_file(self, file_id: str) -> tuple[bytes, str, str]:
        """
        file_id로 파일 다운로드 (읽기 전용)

        Args:
            file_id: Google Drive 파일 ID

        Returns:
            (file_bytes, filename, mime_type) 튜플
        """
        self._ensure_initialized()

        # 파일 메타데이터 조회
        info = self.service.files().get(
            fileId=file_id,
            fields='id, name, mimeType'
        ).execute()

        filename = info['name']
        mime_type = info['mimeType']

        logger.info(f"[RAG AGENT] [DOWNLOAD] 파일 다운로드 시작: {filename} ({mime_type})")

        # Google Docs 형식인 경우 PDF로 export
        google_export_mapping = {
            'application/vnd.google-apps.document': 'application/pdf',
            'application/vnd.google-apps.spreadsheet': 'application/pdf',
            'application/vnd.google-apps.presentation': 'application/pdf',
        }

        if mime_type in google_export_mapping:
            export_mime = google_export_mapping[mime_type]
            request = self.service.files().export_media(fileId=file_id, mimeType=export_mime)
            mime_type = export_mime
        else:
            request = self.service.files().get_media(fileId=file_id)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        file_bytes = buffer.getvalue()
        logger.info(f"[RAG AGENT] [DOWNLOAD] 파일 다운로드 완료: {filename}, {len(file_bytes)} bytes")

        return file_bytes, filename, mime_type


_gdrive_client = _GDriveReadOnlyClient()


def download_by_storage_ref(storage_ref: str) -> tuple[bytes, str, str]:
    """
    storage_ref 주소('gdrive://file/abc123')로 파일을 다운로드합니다.

    Args:
        storage_ref: 저장소 참조 문자열 (예: gdrive://file/abc123)

    Returns:
        (file_bytes, filename, mime_type)

    Raises:
        ValueError: 지원하지 않는 프로토콜인 경우
    """
    if storage_ref.startswith("gdrive://file/"):
        file_id = storage_ref.replace("gdrive://file/", "")
        return _gdrive_client.download_file(file_id)
    elif storage_ref.startswith("gdrive://"):
        file_id = storage_ref.replace("gdrive://", "")
        return _gdrive_client.download_file(file_id)
    else:
        raise ValueError(f"지원하지 않는 storage_ref 프로토콜: {storage_ref}")


# ============================================================================
# State 정의 (RAG 그래프의 공유 업무 메모장)
# ============================================================================

class RAGState(TypedDict):
    """RAG 에이전트 그래프가 노드 간에 주고받는 상태(State) 양식"""
    intent: str                      # 분류된 의도: 'search'(검색) 또는 'index'(인덱싱)
    raw_input: str                   # 사용자가 보낸 원본 입력 텍스트
    question: str                    # 정제된 검색 질문
    search_type: str                 # 검색 방식: 'vector'(의미 검색) 또는 'sql'(메타데이터 검색)
    search_results: List[Dict]       # DB에서 찾은 검색 결과 조각들
    sources: List[Dict]              # 출처 정보 (파일명, storage_ref 등)
    index_request: Optional[Dict]    # 인덱싱할 파일 목록 데이터
    index_result: Optional[Dict]     # 인덱싱 처리 결과 요약
    answer: str                      # 최종 생성된 답변


# ============================================================================
# Intent Router용 구조화된 출력 스키마 (Pydantic)
# ============================================================================

class FileInfo(BaseModel): # [ 1 ]
    """인덱싱할 개별 파일 정보"""
    filename: str = Field(description="파일명")
    storage_ref: str = Field(description="Google Drive 파일 참조 (gdrive://file/xxx 형식)")


class IntentClassification(BaseModel): # [ 2 ]
    """
    사용자 요청의 의도 분류 결과 양식

    - search: 문서 검색, 질문 답변, 정보 조회
    - index: 문서 인덱싱, 벡터 DB 저장, 등록
    """
    intent: Literal["search", "index"] = Field(description="search: 문서 검색, index: 문서 인덱싱")
    question: Optional[str] = Field(default=None, description="search일 때 검색 질문")
    files: Optional[List[FileInfo]] = Field(default=None, description="index일 때 인덱싱할 파일 목록")


class SearchTypeClassification(BaseModel): # [ 3 ]
    """
    검색 방식 분류 결과 양식

    - vector: 문서 내용/의미 기반 벡터 검색
    - sql: 파일명/날짜/형식 등 메타데이터 조건 검색
    """
    search_type: Literal["vector", "sql"] = Field(description="vector: 의미 기반 검색, sql: 메타데이터 조건 검색")


# ============================================================================
# 노드 함수들 (각 부서 담당자)
# ============================================================================

def intent_router(state: RAGState) -> RAGState:
    """
    [1단계 노드] 자연어 입력을 분석하여 인텐트 결정 (검색 vs 인덱싱)

    특히 "하나만 인덱싱해줘", "전부 인덱싱해줘" 같은 사용자의 수량 조건을 스스로 파악하여
    정확히 원하는 파일만 골라냅니다!
    """
    logger.info("[RAG AGENT] [INTENT ROUTER] 시작")
    raw_input = state.get("raw_input", "") # [ 1 ]

    messages = [
        {
            "role": "system",
            "content": """사용자 요청을 분석하여 의도를 파악하고 필요한 정보를 추출하세요.

## 의도 분류
1. search - 문서 검색, 질문 답변, 정보 조회
2. index - 문서 인덱싱, 저장, 등록 요청

## 중요: 파일 선택 규칙 (index인 경우)
요청에 여러 파일이 있을 때, 사용자 의도에 따라 선택하세요:
- "하나만", "첫 번째만", "한 개" → 첫 번째 파일 1개만 선택
- "3개만", "세 개" → 처음 3개만 선택
- "전부", "모든", "다" → 모든 파일 선택
- 수량 지정 없음 → 모든 파일 선택

파일 정보는 "filename (gdrive://file/xxx)" 형식에서 추출하세요.

## 응답 규칙
- search: question 필드에 검색 질문 작성, files는 null
- index: files 배열에 선택된 파일 정보 작성, question은 null"""
        },
        {
            "role": "user",
            "content": raw_input
        }
    ]

    structured_llm = llm.with_structured_output(IntentClassification) # [ 2 ]
    result = structured_llm.invoke(messages)

    if result.intent == "index": # [ 3 ]
        files = result.files or []
        if files:
            files_dict = [f.model_dump() for f in files]
            logger.info(f"[RAG AGENT] [INTENT ROUTER] INDEX: {len(files_dict)}개 파일 선택됨")
            return {
                "intent": INTENT_INDEX,
                "index_request": {"files": files_dict},
                "question": "",
            }
        else:
            logger.warning(f"[RAG AGENT] [INTENT ROUTER] INDEX 의도지만 files가 비어있음")
            return {
                "intent": INTENT_SEARCH,
                "question": raw_input,
                "index_request": None,
            }
    else:
        question = result.question or raw_input # [ 4 ]
        logger.info(f"[RAG AGENT] [INTENT ROUTER] SEARCH: {question[:50]}...")
        return {
            "intent": INTENT_SEARCH,
            "question": question,
            "index_request": None,
        }


def search_router(state: RAGState) -> RAGState:
    """
    [2단계 노드 - 검색 분기] 질문을 보고 '의미 기반 벡터 검색(vector)'으로 갈지,
    '파일 속성 기반 SQL 검색(sql)'으로 갈지 결정합니다.
    """
    logger.info("[RAG AGENT] [SEARCH ROUTER] 시작")
    question = state["question"]

    messages = [
        {
            "role": "system",
            "content": """사용자 질문을 분석하여 적절한 검색 방식을 선택하세요.

검색 방식:
1. vector - 의미 기반 검색 (기본값, 대부분 질문에 적합)
   - 내용 검색: "AI 트렌드", "휴가 규정", "보안 정책"
   - 파일 위치/출처 요청: "원본 파일 위치", "문서 찾아줘"
   - 정보 조회: "~에 대해 알려줘", "~가 뭐야?"

2. sql - 메타데이터 조건 검색 (내용이 아닌 속성으로 필터할 때)
   - 파일 형식 필터: "PDF 문서만", "Excel 파일 목록"
   - 파일명 검색: "SPRi 파일 목록", "Brief 문서들"
   - 날짜 필터: "오늘 인덱싱된 문서", "이번 주 문서", "12월 문서"
   - 목록 조회: "인덱싱된 모든 파일", "문서 목록"

중요: 문서 내용에 대한 질문은 항상 vector를 선택하세요."""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    structured_llm = llm.with_structured_output(SearchTypeClassification)
    result: SearchTypeClassification = structured_llm.invoke(messages)
    logger.info(f"[RAG AGENT] [SEARCH ROUTER] 검색 방식: {result.search_type}")
    return {"search_type": result.search_type}


def vector_search(state: RAGState) -> RAGState:
    """
    [검색 노드 A] 벡터 유사도 검색 (의미 기반)
    질문을 임베딩 벡터로 바꾼 뒤 Supabase의 `match_documents` 함수를 호출하여
    의미가 가장 가까운 문서 청크 5개를 찾아옵니다.
    """
    logger.info("[RAG AGENT] [VECTOR SEARCH] 시작")
    question = state["question"] # [ 1 ]

    query_embedding = get_embedding(question)

    sb = get_supabase()

    try:
        # Supabase pgvector의 match_documents RPC 함수 호출
        response = sb.rpc("match_documents", { # [ 2 ]
            "query_embedding": query_embedding,
            "match_count": 5
        }).execute()

        results = response.data or []
    except Exception as e:
        logger.error(f"[RAG AGENT] [ERROR] 벡터 검색 오류: {e}")
        results = []

    logger.info(f"[RAG AGENT] [VECTOR SEARCH] 검색 결과: {len(results)}개")

    sources = [ # [ 3 ]
        {
            "storage_ref": r.get("storage_ref", ""),
            "filename": r.get("filename", ""),
            "chunk_index": r.get("chunk_index", 0),
            "similarity": r.get("similarity", 0)
        }
        for r in results
    ]

    return {"search_results": results, "sources": sources}


def _get_today_iso() -> str: # [ 1 ]
    from datetime import date
    return date.today().isoformat()


def sql_search(state: RAGState) -> RAGState:
    """
    [검색 노드 B] SQL 조건 검색 (메타데이터 필터)
    파일 형식(pdf, txt), 파일명 키워드, 생성 날짜 조건 등을 추출하여 DB 테이블을 조회합니다.
    """
    logger.info("[RAG AGENT] [SQL SEARCH] 시작")
    question = state["question"] # [ 2 ]

    messages = [
        {
            "role": "user",
            "content": f"""다음 질문에서 검색 조건을 JSON으로 추출하세요.
질문: {question}

사용 가능한 필드:
1. document_type: 파일 형식 (허용 값: pdf, text, markdown, docx, doc, xlsx, xls)
2. filename_contains: 파일명에 포함된 키워드 (예: "SPRi", "Brief", "AI")
3. created_after: 이 날짜 이후 인덱싱된 문서 (ISO 형식: "YYYY-MM-DD")
4. created_before: 이 날짜 이전 인덱싱된 문서 (ISO 형식: "YYYY-MM-DD")
5. list_all: true면 모든 문서 목록 반환

예시:
- "PDF 문서만" -> {{"document_type": "pdf"}}
- "파일명에 AI가 들어간 파일 목록" -> {{"filename_contains": "AI"}}
- "인덱싱된 모든 파일" -> {{"list_all": true}}
- "2025년 12월 문서" -> {{"created_after": "2025-12-01", "created_before": "2025-12-31"}}

참고: 오늘 날짜는 {_get_today_iso()}입니다.

조건을 추출할 수 없으면 빈 객체 {{}}를 반환하세요.
JSON만 응답:"""
        }
    ]

    try:
        response = llm.invoke(messages) # [ 3 ]
        filters = json.loads(response.content.strip())
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[RAG AGENT] [ERROR] 조건 추출 실패: {e}")
        filters = {}

    logger.info(f"[RAG AGENT] [SQL SEARCH] 추출된 조건: {filters}")

    sb = get_supabase()

    try: # [ 4 ]
        query = sb.table("documents").select("storage_ref, filename, document_type, created_at, content")

        # document_type 필터
        if filters.get("document_type"):
            query = query.eq("document_type", filters["document_type"])

        # filename 키워드 검색 (ILIKE)
        if filters.get("filename_contains"):
            query = query.ilike("filename", f"%{filters['filename_contains']}%")

        # created_at 날짜 필터
        if filters.get("created_after"):
            query = query.gte("created_at", filters["created_after"])
        if filters.get("created_before"):
            query = query.lte("created_at", filters["created_before"])

        # 기본: chunk_index=0 (파일당 대표 1개 행)이거나 list_all=true면 전체
        if not filters.get("list_all"):
            query = query.eq("chunk_index", 0)

        resp = query.order("created_at", desc=True).limit(20).execute()
        results = resp.data or []
    except Exception as e:
        logger.error(f"[RAG AGENT] [ERROR] SQL 검색 오류: {e}")
        results = []

    logger.info(f"[RAG AGENT] [SQL SEARCH] 검색 결과: {len(results)}개")

    sources = [ # [ 5 ]
        {
            "storage_ref": r.get("storage_ref", ""),
            "filename": r.get("filename", ""),
            "document_type": r.get("document_type", ""),
            "created_at": r.get("created_at", "")
        }
        for r in results
    ]

    return {"search_results": results, "sources": sources}


def generate(state: RAGState) -> RAGState:
    """
    [답변 생성 노드] 검색된 문서 조각(search_results)과 출처(sources)를 바탕으로
    최종 답변을 작성합니다.
    """
    logger.info("[RAG AGENT] [GENERATE] 시작")
    question = state["question"]
    search_type = state.get("search_type", "vector")
    search_results = state.get("search_results", [])
    sources = state.get("sources", [])

    if not search_results:
        return {
            "answer": "관련 문서를 찾을 수 없습니다. 다른 검색어로 시도해주세요.",
            "sources": []
        }

    # SQL 검색 결과일 때: 깔끔한 문서 목록 형태로 응답
    if search_type == "sql":
        return _generate_list_response(search_results, sources)

    # Vector 검색 결과일 때: 문서 내용을 읽고 gemini-3.8-flash가 질문에 답변 생성
    return _generate_content_response(question, search_results, sources)


def _generate_list_response(search_results: List[Dict], sources: List[Dict]) -> RAGState:
    """
    SQL 검색 결과를 마크다운 목록 형태로 포맷팅합니다.
    """
    logger.info("[RAG AGENT] [GENERATE] 목록 응답 생성")

    answer_parts = [f"**검색 결과: {len(search_results)}개 문서**\n"]

    seen_files = set()
    for i, s in enumerate(sources, 1):
        filename = s.get("filename", "unknown")
        if filename in seen_files:
            continue
        seen_files.add(filename)

        storage_ref = s.get("storage_ref", "")
        doc_type = s.get("document_type", "")
        created_at = s.get("created_at", "")

        # 날짜 포맷팅
        if created_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                created_at = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass

        answer_parts.append(f"{i}. **{filename}**")
        if doc_type:
            answer_parts.append(f"   - 형식: {doc_type}")
        if created_at:
            answer_parts.append(f"   - 인덱싱: {created_at}")
        if storage_ref:
            gdrive_url = _storage_ref_to_url(storage_ref)
            if gdrive_url:
                answer_parts.append(f"   - 위치: {gdrive_url}")
        answer_parts.append("")

    return {
        "answer": "\n".join(answer_parts),
        "sources": sources
    }


def _generate_content_response(question: str, search_results: List[Dict], sources: List[Dict]) -> RAGState:
    """
    Vector 검색 결과를 참고 문서로 넣어 gemini-3.8-flash가 정확한 답변과 출처 링크를 생성합니다.
    """
    logger.info("[RAG AGENT] [GENERATE] 내용 답변 생성")

    context_parts = [] # [ 1 ]
    for r in search_results:
        filename = r.get("filename", "unknown")
        content = r.get("content", "")
        context_parts.append(f"[{filename}]\n{content}")

    context = "\n\n---\n\n".join(context_parts)

    messages = [ # [ 2 ]
        {
            "role": "user",
            "content": f"""당신은 사내 문서 기반 RAG 전문가입니다.
아래 참고 문서를 바탕으로 질문에 정확하게 답변하세요.

## 참고 문서
{context}

## 질문
{question}

## 지시사항
- 문서에 있는 정보만 사용하세요
- 출처 문서명을 명시하세요
"""
        }
    ]

    generate_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
    response = generate_llm.invoke(messages)
    answer = response.content

    # 하단에 원본 Google Drive 바로가기 링크(출처) 자동 첨부!
    source_info = "\n\n**출처:**\n" # [ 3 ]
    seen = set()
    for s in sources:
        filename = s.get("filename", "")
        storage_ref = s.get("storage_ref", "")
        if filename and filename not in seen:
            source_info += f"- {filename}"
            if storage_ref:
                # storage_ref를 클릭 가능한 Google Drive 웹 URL로 변환
                gdrive_url = _storage_ref_to_url(storage_ref)
                if gdrive_url:
                    source_info += f"\n  파일 위치: {gdrive_url}"
                else:
                    source_info += f" (`{storage_ref}`)"
            source_info += "\n"
            seen.add(filename)

    return {"answer": answer + source_info, "sources": sources}


def _storage_ref_to_url(storage_ref: str) -> Optional[str]:
    """
    내부 참조 주소(`gdrive://file/xxx`)를 사용자가 브라우저에서 바로 열 수 있는
    Google Drive 웹 URL(`https://drive.google.com/file/d/xxx/view`)로 변환합니다.
    """
    if not storage_ref:
        return None

    file_id = None
    if storage_ref.startswith("gdrive://file/"):
        file_id = storage_ref.replace("gdrive://file/", "")
    elif storage_ref.startswith("gdrive://"):
        file_id = storage_ref.replace("gdrive://", "")

    if file_id:
        return f"https://drive.google.com/file/d/{file_id}/view"
    return None


def index_document_node(state: RAGState) -> RAGState:
    """
    [인덱싱 노드] 선택된 파일들을 Google Drive에서 다운로드하여 텍스트를 추출하고,
    500자 단위로 쪼갠(Chunking) 뒤 임베딩하여 Supabase DB에 저장합니다.
    """
    logger.info("[RAG AGENT] [INDEX] 시작")

    index_request = state.get("index_request", {}) # [ 1 ]
    logger.info(f"[RAG AGENT] [INDEX] index_request keys: {list(index_request.keys()) if index_request else 'None'}")

    if not index_request:
        return {
            "index_result": {"status": "error", "message": "인덱싱 요청 데이터가 없습니다."},
            "answer": "인덱싱 요청 데이터가 없습니다."
        }

    files_to_index = index_request.get("files", []) # [ 2 ]

    if not files_to_index:
        return {
            "index_result": {"status": "error", "message": "인덱싱할 파일이 없습니다."},
            "answer": "인덱싱할 파일이 없습니다."
        }

    logger.info(f"[RAG AGENT] [INDEX] {len(files_to_index)}개 파일 처리")

    sb = get_supabase() # [ 3 ]
    all_results = []

    for file_info in files_to_index:
        storage_ref = file_info.get("storage_ref", "")
        filename = file_info.get("filename", "")

        logger.info(f"[RAG AGENT] [INDEX] 파일 처리 중: {filename} ({storage_ref})")

        try:
            result = _index_single_file(sb, storage_ref, filename)
            all_results.append(result)
            logger.info(f"[RAG AGENT] [INDEX] 완료: {filename}, 청크 {result['chunk_count']}개")
        except Exception as e:
            logger.error(f"[RAG AGENT] [INDEX] [ERROR] 실패: {filename}, 오류: {e}")
            all_results.append({
                "status": "error",
                "filename": filename,
                "storage_ref": storage_ref,
                "error": str(e)
            })

    # 결과 요약
    success_count = sum(1 for r in all_results if r.get("status") == "success") # [ 4 ]

    answer_parts = [f"✅ 문서 인덱싱 완료 ({success_count}/{len(all_results)}개 성공)"]
    for r in all_results:
        if r.get("status") == "success":
            answer_parts.append(f"  {r['filename']}: {r['chunk_count']}개 청크")
        else:
            answer_parts.append(f"  {r.get('filename', 'unknown')}: {r.get('error', '알 수 없는 오류')}")

    return {
        "index_result": {"status": "success", "results": all_results},
        "answer": "\n".join(answer_parts)
    }


def _index_single_file(sb, storage_ref: str, filename: str) -> dict:
    """
    단일 파일 인덱싱 헬퍼 함수:
    1. storage_ref로 파일 다운로드 -> 2. 텍스트 추출 -> 3. 청킹(500자) -> 4. 임베딩 & DB 저장
    """
    file_bytes, downloaded_filename, mime_type = download_by_storage_ref(storage_ref) # [ 1 ]
    filename = downloaded_filename

    content = extract_text_from_bytes(file_bytes, mime_type)
    logger.info(f"[RAG AGENT] [INDEX] 텍스트 추출: {filename}, {len(content)}자")

    if not content:
        raise ValueError("텍스트 추출 결과가 비어있습니다.")

    # MIME 타입에서 document_type 자동 추론
    mime_to_doctype = { # [ 2 ]
        "application/pdf": "pdf",
        "text/plain": "text",
        "text/markdown": "markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/msword": "doc",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-excel": "xls",
    }

    document_type = mime_to_doctype.get(mime_type)
    if not document_type and filename and '.' in filename:
        document_type = filename.rsplit('.', 1)[-1].lower()

    # 긴 문서를 500자씩(50자 겹침) 잘라서 검색 정확도를 높입니다.
    text_splitter = RecursiveCharacterTextSplitter( # [ 3 ]
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_text(content)
    logger.info(f"[RAG AGENT] [INDEX] 청크 수: {len(chunks)}")

    # 각 청크를 벡터로 임베딩하여 Supabase 'documents' 테이블에 저장
    for i, chunk in enumerate(chunks): # [ 4 ]
        embedding = get_embedding(chunk)

        data = {
            "content": chunk,
            "embedding": embedding,
            "filename": filename,
            "storage_ref": storage_ref,
            "chunk_index": i,
        }

        if document_type:
            data["document_type"] = document_type

        sb.table("documents").insert(data).execute()

    return { # [ 5 ]
        "status": "success",
        "filename": filename,
        "storage_ref": storage_ref,
        "chunk_count": len(chunks)
    }


# ============================================================================
# 조건부 엣지를 위한 라우팅 함수 (갈림길 안내원)
# ============================================================================

def route_intent(state: RAGState) -> Literal["search_router", "index_document_node"]: # [ 1 ]
    """인텐트('search' vs 'index')에 따라 검색 또는 인덱싱 노드로 분기"""
    intent = state.get("intent", INTENT_SEARCH)

    if intent == INTENT_INDEX:
        return "index_document_node"
    else:
        return "search_router"


def route_search(state: RAGState) -> Literal["vector_search", "sql_search"]: # [ 2 ]
    """검색 방식('vector' vs 'sql')에 따라 분기"""
    search_type = state.get("search_type", "vector")

    if search_type == "sql":
        return "sql_search"
    else:
        return "vector_search"


# ============================================================================
# LangGraph 그래프 생성 및 조립
# ============================================================================

def create_rag_graph():
    """
    RAG 에이전트 상태 그래프(StateGraph) 생성 및 컴파일

    그래프 구조:

        START
          │
          ▼
      intent_router
          │
          ├──────────────────┐
          │                  │
          ▼                  ▼
    search_router     index_document_node
          │                  │
       ┌──┴──┐              │
       │     │              │
       ▼     ▼              │
    vector  sql             │
       │     │              │
       └──┬──┘              │
          │                  │
          ▼                  │
       generate              │
          │                  │
          ├──────────────────┘
          │
          ▼
         END
    """

    graph_builder = StateGraph(RAGState)

    # 1. 6개 작업 노드 등록
    graph_builder.add_node("intent_router", intent_router)
    graph_builder.add_node("search_router", search_router)
    graph_builder.add_node("vector_search", vector_search)
    graph_builder.add_node("sql_search", sql_search)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("index_document_node", index_document_node)

    # 2. 출발점 및 조건부 갈림길(엣지) 연결
    graph_builder.add_edge(START, "intent_router")

    graph_builder.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "search_router": "search_router",
            "index_document_node": "index_document_node"
        }
    )

    graph_builder.add_conditional_edges(
        "search_router",
        route_search,
        {
            "vector_search": "vector_search",
            "sql_search": "sql_search"
        }
    )

    # 3. 검색 후 답변 생성 및 종료 연결
    graph_builder.add_edge("vector_search", "generate")
    graph_builder.add_edge("sql_search", "generate")
    graph_builder.add_edge("generate", END)
    graph_builder.add_edge("index_document_node", END)

    return graph_builder.compile()


graph = create_rag_graph()


class InternalRAGAgent:
    """
    Internal RAG 에이전트 클래스

    독립적인 LangGraph 에이전트로 실행 가능하며,
    A2A 프로토콜 실행자(InternalRAGAgentExecutor)에서도 호출됩니다.
    """

    # 각 노드를 지날 때 사용자에게 보여줄 친절한 진행 상태 메시지
    NODE_MESSAGES = { # [ 1 ]
        "intent_router": "🎯 요청 분석 중...",
        "search_router": "🔍 검색 방식 결정 중...",
        "vector_search": "📊 벡터 유사도 검색 중...",
        "sql_search": "🗃️ 메타데이터 검색 중...",
        "generate": "📝 답변 생성 중...",
        "index_document_node": "📥 문서 인덱싱 중...",
    }

    def __init__(self):
        self.graph = create_rag_graph()

    async def stream(
        self,
        query: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        LangGraph 스트리밍 실행: 각 노드가 완료될 때마다 진행 상황을 실시간으로 중계합니다.
        """
        inputs = {"raw_input": query} # [ 2 ]
        final_state = None

        async for event in self.graph.astream(inputs, stream_mode="updates"):
            for node_name, node_output in event.items():
                message = self.NODE_MESSAGES.get(node_name, f"처리 중: {node_name}")
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": message,
                }

                if node_name in ("generate", "index_document_node"):
                    final_state = node_output

        answer = final_state.get("answer", "처리 결과가 없습니다.") if final_state else "처리 결과가 없습니다." # [ 3 ]
        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": answer,
        }


if __name__ == "__main__":
    import asyncio

    async def test():
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
            with open("rag_graph.png", "wb") as f:
                f.write(png_bytes)
            print("그래프 이미지 저장: rag_graph.png")
        except Exception as e:
            print(f"(그래프 시각화 생략: {e})")

        print("\n" + "=" * 60)
        print("Internal RAG Agent 테스트")
        print("=" * 60)

        agent = InternalRAGAgent()
        query = "AI 트렌드 관련 내용이 담긴 문서 찾아줘"

        print(f"\n👤 Query: {query}")
        print("-" * 60)

        async for chunk in agent.stream(query):
            content = chunk.get("content", "")
            is_complete = chunk.get("is_task_complete", False)

            if not is_complete:
                print(f"{content}")
            else:
                print(f"\n💬 {content}")

    asyncio.run(test())
