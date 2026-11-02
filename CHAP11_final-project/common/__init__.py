# ==============================================================================
# 📘 [CHAP11] 실전 파이널 프로젝트 - A2A 공통 모듈 패키지 (__init__.py)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - 이 파일은 4개 에이전트(오케스트레이터, 웹 리서치, 사내 RAG, 파일 관리)가
#   서로 대화할 때 공통으로 쓰는 '표준 서류 양식(schemas)', '사내 전화기(a2a_client)',
#   '부서별 내선번호 안내판(config)'을 한곳에 모아둔 공용 문구함입니다!
# ==============================================================================

# A2A 공통 모듈
from .schemas import (
    # A2A SDK 네이티브 타입 (re-export: 외부에서도 쉽게 꺼내 쓸 수 있게 재노출)
    Task,
    TaskState,
    TaskStatus,
    Artifact,
    Part,
    TextPart,
    DataPart,
    FilePart,
    FileWithUri,
    FileWithBytes,
    Message,
    new_task,
    new_agent_text_message,
    # 비즈니스 로직용 커스텀 타입
    TaskIntent,
    ArtifactType,
    AgentContext,
    AgentRequest,
    AgentResponse,
    # 아티팩트(결과물 상자) 생성 및 추출 헬퍼 함수
    create_text_artifact,
    create_data_artifact,
    create_file_artifact,
    create_file_bytes_artifact,
    get_artifact_text,
    get_artifact_data,
    get_artifact_file_uri,
    get_artifact_file_bytes,
)
from .a2a_client import A2AClientWrapper
from .config import AgentConfig, get_agent_urls

__all__ = [
    # A2A SDK 네이티브 타입
    'Task',
    'TaskState',
    'TaskStatus',
    'Artifact',
    'Part',
    'TextPart',
    'DataPart',
    'FilePart',
    'FileWithUri',
    'FileWithBytes',
    'Message',
    'new_task',
    'new_agent_text_message',
    # 비즈니스 로직용 타입
    'TaskIntent',
    'ArtifactType',
    'AgentContext',
    'AgentRequest',
    'AgentResponse',
    # 헬퍼 함수
    'create_text_artifact',
    'create_data_artifact',
    'create_file_artifact',
    'create_file_bytes_artifact',
    'get_artifact_text',
    'get_artifact_data',
    'get_artifact_file_uri',
    'get_artifact_file_bytes',
    # 클라이언트/설정
    'A2AClientWrapper',
    'AgentConfig',
    'get_agent_urls',
]
