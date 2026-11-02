# ==============================================================================
# 📘 [CHAP11] 실전 파이널 프로젝트 - 에이전트 설정 및 포트 안내판 (config.py)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - 사내 4개 부서(오케스트레이터, 웹 리서치, 사내 RAG, 파일 관리)가 각각 몇 층 몇 호(포트 번호)에
#   입주해 있는지 적어둔 '사내 부서 배치도(내선번호 안내판)'입니다!
#   * 10010번 포트: Orchestrator Agent (총괄 PM 부서)
#   * 10011번 포트: Web Research Agent (외부 시장조사/검색 부서)
#   * 10012번 포트: Internal RAG Agent (사내 지식도서관/DB 부서)
#   * 10013번 포트: File Management Agent (문서 보관소/구글 드라이브 관리 부서)
# ==============================================================================

"""
에이전트 설정 및 URL 관리
"""
from typing import Dict
from pydantic import BaseModel
import os


class AgentConfig(BaseModel):
    """개별 에이전트 서버의 기본 설정 정보"""
    name: str
    url: str
    port: int
    description: str


# 4개 핵심 에이전트의 URL 및 포트 설정
AGENT_CONFIGS: Dict[str, AgentConfig] = {
    "orchestrator": AgentConfig(
        name="Orchestrator Agent",
        url="http://localhost:10010",
        port=10010,
        description="사용자 요청을 분석하고 작업을 조율하는 중앙 오케스트레이터"
    ),
    "web_research": AgentConfig(
        name="Web Research Agent",
        url="http://localhost:10011",
        port=10011,
        description="MCP 기반 외부 웹 검색 및 정보 수집 에이전트"
    ),
    "internal_rag": AgentConfig(
        name="Internal RAG Agent",
        url="http://localhost:10012",
        port=10012,
        description="사내 문서/DB 기반 RAG 검색 및 답변 생성 에이전트"
    ),
    "file_management": AgentConfig(
        name="File Management Agent",
        url="http://localhost:10013",
        port=10013,
        description="Google Drive 기반 파일 관리 에이전트"
    ),
}


def get_agent_urls() -> Dict[str, str]:
    """모든 에이전트의 {이름: URL} 딕셔너리 반환"""
    return {name: config.url for name, config in AGENT_CONFIGS.items()}


def get_agent_config(agent_name: str) -> AgentConfig:
    """특정 에이전트의 상세 설정 반환"""
    if agent_name not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent: {agent_name}")
    return AGENT_CONFIGS[agent_name]
