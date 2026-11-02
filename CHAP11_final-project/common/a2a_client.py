# ==============================================================================
# 📘 [CHAP11] 실전 파이널 프로젝트 - A2A 클라이언트 래퍼 (A2AClientWrapper)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - A2AClientWrapper = 사내 각 부서(전문 에이전트)에 한 번에 연락할 수 있는 '스마트 사내 전화기'
#   초기화할 때 각 부서의 명함(AgentCard)을 자동으로 수집해 단축번호로 등록해 두고,
#   `send_message("web_research", "AI 트렌드 조사해줘")`처럼 부서 이름만 부르면
#   A2A 표준 봉투에 담아 전송하고 결과물 상자(Artifact)를 깔끔하게 정리해서 돌려줍니다!
# ==============================================================================

import asyncio
import json
import logging
from uuid import uuid4
from typing import Optional, Dict, Any, List
import httpx

logger = logging.getLogger(__name__)

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    Message,
    MessageSendParams,
    SendMessageRequest,
    Part,
    TextPart,
    DataPart,
    TaskState,
)

from .schemas import (
    AgentResponse,
    get_artifact_text,
    get_artifact_data,
    get_artifact_file_uri,
)


class A2AClientWrapper:
    """
    A2A 클라이언트 래퍼 (스마트 사내 직통 전화기)

    복잡한 A2A SDK 메시지 조립과 응답 파싱 과정을 감싸서(Wrapping),
    에이전트 간 통신을 몇 줄의 코드로 쉽게 수행할 수 있도록 지원합니다.
    """

    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout
        self.httpx_client: Optional[httpx.AsyncClient] = None
        self.agents: Dict[str, Dict[str, Any]] = {}

    async def initialize(self, agent_urls: Dict[str, str]) -> None:
        """
        에이전트 클라이언트 초기화 (각 부서 에이전트와 명함 교환하기)

        Args:
            agent_urls: 에이전트 이름과 URL 매핑 딕셔너리
        """
        self.httpx_client = httpx.AsyncClient(timeout=self.timeout)

        for name, url in agent_urls.items():
            try:
                # 1. 해당 URL의 에이전트에게 명함(AgentCard) 요청
                card_resolver = A2ACardResolver(
                    httpx_client=self.httpx_client,
                    base_url=url,
                )
                agent_card = await card_resolver.get_agent_card()

                # 2. 명함을 기반으로 전용 A2A 클라이언트 생성
                client = A2AClient(
                    httpx_client=self.httpx_client,
                    agent_card=agent_card,
                )

                self.agents[name] = {
                    'card': agent_card,
                    'client': client,
                    'url': url,
                }

                logger.info(f"[A2A CLIENT] {name} 에이전트 연결: {agent_card.name}")

            except Exception as e:
                logger.error(f"[A2A CLIENT] [ERROR] {name} 에이전트 연결 실패: {str(e)}")

    async def close(self) -> None:
        """클라이언트 종료 (통신 회선 정리)"""
        if self.httpx_client:
            await self.httpx_client.aclose()

    async def send_message(
        self,
        agent_name: str,
        message: str,
        blocking: bool = True
    ) -> AgentResponse:
        """
        A2A 메시지를 대상 에이전트에게 전송하고 정리된 응답(AgentResponse)을 받습니다.

        Args:
            agent_name: 대상 에이전트 이름 (예: 'web_research', 'internal_rag', 'file_management')
            message: 전송할 메시지 (자연어 텍스트 또는 JSON 문자열)
            blocking: True면 작업 완료까지 대기

        Returns:
            AgentResponse: 에이전트 응답 (성공 여부, 추출된 artifacts 목록 포함)
        """
        if agent_name not in self.agents:
            available = ", ".join(self.agents.keys())
            return AgentResponse(
                success=False,
                error=f"에이전트 '{agent_name}'를 찾을 수 없습니다. 사용 가능: {available}"
            )

        agent = self.agents[agent_name]
        client = agent['client']

        try:
            # 1. A2A Message(편지 내용) 생성
            a2a_message = Message(
                role='user',
                parts=[Part(root=TextPart(text=message))],
                message_id=uuid4().hex,
            )

            # 2. SendMessageRequest(편지 봉투) 생성
            request = SendMessageRequest(
                id=uuid4().hex,
                params=MessageSendParams(message=a2a_message),
            )

            # 3. A2A 요청 전송 및 응답 대기
            response = await asyncio.wait_for(
                client.send_message(request),
                timeout=self.timeout
            )

            # 4. 응답 상자 개봉
            result = response.root.result

            # Case A: Task(작업) 응답이며 결과물 상자(artifacts)가 들어있는 경우
            if hasattr(result, 'artifacts') and result.artifacts:
                artifacts_data = []
                for artifact in result.artifacts:
                    artifact_dict = {
                        "artifact_id": artifact.artifact_id,
                        "name": artifact.name,
                        "description": artifact.description,
                    }
                    # 텍스트(TextPart) 추출
                    text = get_artifact_text(artifact)
                    if text:
                        artifact_dict["text"] = text
                    # 구조화 데이터(DataPart) 추출
                    data = get_artifact_data(artifact)
                    if data:
                        artifact_dict["data"] = data
                    # 파일 주소(FilePart URI) 추출
                    file_uri = get_artifact_file_uri(artifact)
                    if file_uri:
                        artifact_dict["file_uri"] = file_uri

                    artifacts_data.append(artifact_dict)

                return AgentResponse(
                    success=True,
                    artifacts=artifacts_data,
                    message="태스크 완료"
                )

            # Case B: 단순 Message 직접 응답인 경우
            if hasattr(result, 'parts') and result.parts:
                texts = []
                for part in result.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        texts.append(part.root.text)
                    elif hasattr(part, 'text'):
                        texts.append(part.text)

                if texts:
                    response_text = "\n".join(texts)
                    return AgentResponse(
                        success=True,
                        artifacts=[{"name": "response", "text": response_text}],
                        message="응답 완료"
                    )

            return AgentResponse(
                success=False,
                error="응답을 받지 못했습니다"
            )

        except asyncio.TimeoutError:
            return AgentResponse(
                success=False,
                error=f"타임아웃 ({self.timeout}초)"
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=str(e)
            )

    async def send_streaming_message(
        self,
        agent_name: str,
        message: str,
    ):
        """
        A2A 스트리밍 메시지 전송 (비동기 제너레이터)
        에이전트가 작업을 처리하는 중간중간 보내는 상태 알림과 최종 결과물을 실시간으로 받아옵니다.

        Args:
            agent_name: 대상 에이전트 이름
            message: 전송할 메시지

        Yields:
            스트리밍 이벤트 딕셔너리 (상태 업데이트 'status' 또는 산출물 'artifact')
        """
        if agent_name not in self.agents:
            yield {"error": f"에이전트 '{agent_name}'를 찾을 수 없습니다"}
            return

        agent = self.agents[agent_name]
        client = agent['client']

        try:
            a2a_message = Message(
                role='user',
                parts=[Part(root=TextPart(text=message))],
                message_id=uuid4().hex,
            )

            request = SendMessageRequest(
                id=uuid4().hex,
                params=MessageSendParams(message=a2a_message),
            )

            # 스트리밍 응답 실시간 수신
            async for event in client.send_message_streaming(request):
                if hasattr(event, 'root'):
                    event_data = event.root

                    # 1. 작업 진행 상태 업데이트 이벤트 (TaskStatusUpdateEvent)
                    if hasattr(event_data, 'status'):
                        yield {
                            "type": "status",
                            "state": event_data.status.state.value if hasattr(event_data.status.state, 'value') else str(event_data.status.state),
                        }

                    # 2. 결과물(아티팩트) 도착 이벤트 (TaskArtifactUpdateEvent)
                    elif hasattr(event_data, 'artifact'):
                        artifact = event_data.artifact
                        artifact_dict = {
                            "type": "artifact",
                            "artifact_id": artifact.artifact_id,
                            "name": artifact.name,
                        }
                        text = get_artifact_text(artifact)
                        if text:
                            artifact_dict["text"] = text
                        data = get_artifact_data(artifact)
                        if data:
                            artifact_dict["data"] = data
                        yield artifact_dict

        except Exception as e:
            yield {"error": str(e)}
