# ==============================================================================
# 📘 [CHAP11] 중앙 오케스트레이터 에이전트 본체 (OrchestratorAgent)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - OrchestratorAgent = 회사 전체 업무를 총괄 지휘하는 '수석 프로젝트 매니저(총괄 PM)'입니다!
# - 고객이 복잡한 요청을 하면 총괄 PM(gemini-3.8-flash)은 다음 4단계를 거칩니다:
#   1) Intent 분석 & Plan 수립 (`analyze_intent`):
#      "이 요청은 1단계로 웹 리서치 팀에 조사를 시키고, 2단계로 파일 관리 팀에 저장을 시켜야겠군!"
#   2) 순차적 외주 실행 & 컨텍스트 체이닝 (`call_remote_agent`):
#      이전 부서가 만든 결과물(예: 웹 검색 텍스트, 또는 구글 드라이브 파일 목록 `DataPart`)을
#      다음 부서(예: 파일 업로드, 또는 RAG 인덱싱)의 입력으로 바통 터치(릴레이)해 줍니다!
#   3) 실패 전파 방지 (`previous_step_failed`):
#      앞 단계 부서 작업이 실패하면 뒷 단계 작업을 무리하게 실행하지 않고 안전하게 건너뜁니다.
#   4) 최종 보고서 작성 (`generate_final_response`):
#      모든 부서의 결과를 종합해 고객에게 친절한 최종 답변을 브리핑합니다!
# ==============================================================================

import os
import json
import logging
import httpx
from typing import AsyncIterator, Dict, Any, List
from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, SendMessageRequest, Part, TextPart, Message
from uuid import uuid4

load_dotenv()


# 산하 3개 전문 에이전트 부서의 주소록
AGENT_URLS = {
    "internal_rag": os.getenv("RAG_AGENT_URL", "http://localhost:10012"),
    "web_research": os.getenv("WEB_AGENT_URL", "http://localhost:10011"),
    "file_management": os.getenv("FILE_AGENT_URL", "http://localhost:10013"),
}


class OrchestratorAgent:
    """
    오케스트레이터 에이전트 (Host Agent / 총괄 PM)

    기능:
    1. Intent 분석 - 어떤 에이전트가 필요한지 판단
    2. Plan 생성 - 멀티스텝 작업 순서 결정 (JSON 모드)
    3. Remote Agent 호출 - A2A Client로 원격 에이전트와 통신 및 DataPart 아티팩트 전달
    4. 결과 통합 - 여러 에이전트의 결과를 하나로 합쳐 최종 답변 생성
    """

    SYSTEM_PROMPT = """당신은 사용자 요청을 분석하고 적절한 에이전트에게 작업을 위임하는 Orchestrator입니다.

사용 가능한 에이전트:
- internal_rag: 사내 문서 검색(RAG), 문서 인덱싱/저장 (storage_ref로 파일 직접 다운로드 후 인덱싱)
- web_research: 외부 웹 검색, 최신 뉴스, 트렌드 정보
- file_management: Google Drive 파일 목록 조회, 검색 (파일 관리 전문)

핵심 원칙:
1. 사용자의 **원래 의도**를 그대로 에이전트에게 전달하세요.
2. 에이전트가 스스로 판단할 수 있도록 자연어로 요청하세요.
3. "하나만", "전부", "3개" 같은 조건도 그대로 전달하세요.

분석 결과를 JSON으로 응답하세요:
{
    "intent": "INTERNAL_SEARCH|WEB_SEARCH|FILE_OPERATION|HYBRID|DIRECT",
    "plan": [
        {"agent": "에이전트명", "query": "사용자 의도를 포함한 자연어 요청"}
    ],
    "direct_answer": "직접 답변 가능한 경우 여기에 작성"
}

=== 핵심: 사용자 의도 전달 ===

잘못된 예 (의도 누락):
사용자: "보고서 폴더 파일 중 하나만 인덱싱해줘"
→ query: "보고서 폴더 파일을 인덱싱해줘"  **("하나만" 누락)**

**올바른 예 (의도 보존):**
사용자: "보고서 폴더 파일 중 하나만 인덱싱해줘"
→ query: "보고서 폴더 파일 중 하나만 인덱싱해줘"  **(원래 의도 유지)**

=== 파일 인덱싱 워크플로우 (2단계) ===

사용자: "보고서 폴더 파일 중 하나만 DB에 인덱싱해줘"
→ plan: [
    {"agent": "file_management", "query": "보고서 폴더의 파일 목록을 검색해줘"},
    {"agent": "internal_rag", "query": "다음 파일들 중 하나만 인덱싱해줘: [파일 목록]"}
  ]

사용자: "reports 폴더 파일 전부 인덱싱해줘"
→ plan: [
    {"agent": "file_management", "query": "reports 폴더의 파일 목록을 검색해줘"},
    {"agent": "internal_rag", "query": "다음 파일들을 모두 인덱싱해줘: [파일 목록]"}
  ]

=== 단일 에이전트 예시 ===
- "휴가 규정 알려줘" → internal_rag
- "오늘 AI 뉴스" → web_research
- "Drive 파일 목록" → file_management
- "안녕" → DIRECT

중요: 에이전트에게 보내는 query에 사용자의 **원래 의도**(수량, 조건 등)를 반드시 포함하세요!
"""

    def __init__(self):
        """초기화: Gemini OpenAI 호환 엔드포인트로 AsyncOpenAI 클라이언트를 설정합니다."""
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY (또는 OPENAI_API_KEY) 환경 변수가 필요합니다")

        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.httpx_client = None
        self.remote_agents: Dict[str, A2AClient] = {}
        self.initialized = False

    async def initialize(self) -> None:
        """산하 3개 Remote Agent의 명함(AgentCard)을 조회하고 A2A 연결을 초기화합니다."""
        if self.initialized:
            return

        self.httpx_client = httpx.AsyncClient(timeout=120.0)

        for name, url in AGENT_URLS.items():
            try:
                # 1. AgentCard(명함) 조회
                card_resolver = A2ACardResolver(
                    httpx_client=self.httpx_client,
                    base_url=url,
                )
                agent_card = await card_resolver.get_agent_card()

                # 2. A2A Client 생성
                client = A2AClient(
                    httpx_client=self.httpx_client,
                    agent_card=agent_card,
                )

                self.remote_agents[name] = client
                logger.info(f"[ORCHESTRATOR] [INIT] {name} 연결 완료: {agent_card.name}")

            except Exception as e:
                logger.error(f"[ORCHESTRATOR] [ERROR] {name} 연결 실패 ({url}): {e}")

        self.initialized = True
        logger.info(f"[ORCHESTRATOR] [INIT] 초기화 완료 (연결된 에이전트: {len(self.remote_agents)}개)")

    async def close(self) -> None:
        """리소스 정리"""
        if self.httpx_client:
            await self.httpx_client.aclose()

    async def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        사용자 질문을 분석하여 실행 계획(Plan)을 JSON 형식으로 생성합니다.

        Args:
            query: 사용자 질문

        Returns:
            {"intent": "...", "plan": [...], "direct_answer": "..."}
        """
        response = await self.openai_client.chat.completions.create(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    async def call_remote_agent(self, agent_name: str, query: str) -> Dict[str, Any]:
        """
        A2A 프로토콜로 원격 전문 에이전트를 호출하고 텍스트 및 데이터 아티팩트를 추출합니다.

        Args:
            agent_name: 호출할 에이전트 이름 ('web_research', 'file_management', 'internal_rag')
            query: 에이전트에게 전달할 자연어 요청

        Returns:
            {"success": bool, "content": str, "artifacts": [...]}
        """
        if agent_name not in self.remote_agents: # [ 1 ]
            return {
                "success": False,
                "content": f"에이전트 '{agent_name}'를 찾을 수 없습니다.",
                "artifacts": []
            }

        client = self.remote_agents[agent_name]

        try: # [ 2 ]
            # A2A 메시지 생성 및 전송
            message = Message(
                kind='message',
                role='user',
                parts=[TextPart(kind='text', text=query)],
                message_id=uuid4().hex,
            )

            request = SendMessageRequest(
                id=uuid4().hex,
                params=MessageSendParams(message=message),
            )

            response = await client.send_message(request)

            # 결과 추출 (response.root.result 구조)
            content = "" # [ 3 ]
            artifacts = []

            result = response.root.result if hasattr(response, 'root') else response.result
            logger.info(f"[ORCHESTRATOR] [REMOTE] {agent_name} result 타입: {type(result)}")

            if result:
                # 1. artifacts 상자에서 TextPart와 DataPart 추출
                if hasattr(result, 'artifacts') and result.artifacts:
                    logger.info(f"[ORCHESTRATOR] [REMOTE] {agent_name} artifacts 수: {len(result.artifacts)}")
                    for artifact in result.artifacts:
                        artifact_data = {
                            "name": artifact.name,
                        }
                        logger.info(f"[ORCHESTRATOR] [REMOTE] artifact.name: {artifact.name}, parts: {len(artifact.parts)}")

                        # Part에서 내용 추출
                        for part in artifact.parts:
                            logger.info(f"[ORCHESTRATOR] [REMOTE] part type: {type(part)}, has root: {hasattr(part, 'root')}")
                            if hasattr(part, 'root'):
                                part_root = part.root
                                logger.info(f"[ORCHESTRATOR] [REMOTE] part.root type: {type(part_root)}")

                                # TextPart 처리 (일반 텍스트 응답)
                                if hasattr(part_root, 'text'):
                                    content = part_root.text
                                    artifact_data["text"] = content
                                    logger.info(f"[ORCHESTRATOR] [REMOTE] TextPart 추출: {content[:100] if content else 'None'}...")

                                # DataPart 처리 (파일 목록 등 구조화 JSON 데이터)
                                elif hasattr(part_root, 'data'):
                                    data = part_root.data
                                    artifact_data["data"] = data
                                    logger.info(f"[ORCHESTRATOR] [REMOTE] DataPart 추출: type={type(data)}")
                                    if isinstance(data, dict):
                                        logger.info(f"[ORCHESTRATOR] [REMOTE] DataPart keys: {list(data.keys())}")
                                        if "files" in data and isinstance(data["files"], list):
                                            logger.info(f"[ORCHESTRATOR] [REMOTE] 파일 목록 발견: {len(data['files'])}개")

                        artifacts.append(artifact_data)
                else:
                    logger.info(f"[ORCHESTRATOR] [REMOTE] {agent_name} artifacts 없음")

                # 2. 만약 artifacts가 비어있다면 대화 기록(history)에서 마지막 에이전트 메시지 추출 (백업)
                if not content and hasattr(result, 'history') and result.history: # [ 4 ]
                    for msg in reversed(result.history):
                        if hasattr(msg, 'role') and 'agent' in str(msg.role):
                            for part in msg.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    content = part.root.text
                                    break
                        if content:
                            break

            logger.info(f"[ORCHESTRATOR] [REMOTE] {agent_name} 최종: content={content[:100] if content else 'None'}..., artifacts={len(artifacts)}")
            return {
                "success": True if content else False,
                "content": content or "응답을 받지 못했습니다.",
                "artifacts": artifacts
            }

        except Exception as e:
            return {
                "success": False,
                "content": f"에이전트 호출 실패: {str(e)}",
                "artifacts": []
            }

    async def generate_final_response(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        여러 에이전트의 작업 결과를 하나로 통합하여 사용자에게 명확한 최종 답변을 생성합니다.
        """
        results_text = json.dumps(results, ensure_ascii=False, indent=2)

        response = await self.openai_client.chat.completions.create(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            messages=[
                {
                    "role": "system",
                    "content": "여러 에이전트의 결과를 통합하여 사용자에게 명확한 답변을 제공하세요."
                },
                {
                    "role": "user",
                    "content": f"""원본 질문: {query}

에이전트 결과:
{results_text}

위 정보를 바탕으로 사용자 질문에 답변해주세요."""
                }
            ],
        )

        return response.choices[0].message.content

    async def stream(
        self,
        query: str,
        session_id: str = "default"
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        오케스트레이터의 전체 워크플로우(분석 -> 계획 -> 릴레이 호출 -> 통합)를 스트리밍으로 실행합니다.
        """
        # 1. 원격 에이전트 연결 초기화
        if not self.initialized: # [ 1 ]
            await self.initialize()

        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "🤔 질문을 분석하고 있습니다..."
        }

        # 2. Intent 분석 및 멀티스텝 Plan 수립
        try: # [ 2 ]
            analysis = await self.analyze_intent(query)
            logger.info(f"[ORCHESTRATOR] Intent 분석 결과: {json.dumps(analysis, ensure_ascii=False, indent=2)}")
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Intent 분석 실패: {e}")
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"분석 중 오류: {str(e)}"
            }
            return

        intent = analysis.get("intent", "DIRECT")
        plan = analysis.get("plan", [])
        direct_answer = analysis.get("direct_answer", "")

        logger.info(f"[ORCHESTRATOR] intent={intent}, plan개수={len(plan)}, remote_agents={list(self.remote_agents.keys())}")

        # Plan 상세 로깅
        for i, step in enumerate(plan): # [ 1 ]
            logger.info(f"[ORCHESTRATOR] Plan[{i}]: agent={step.get('agent')}, query={step.get('query', '')[:100]}...")

        if not plan and intent != "DIRECT":
            logger.warning(f"[ORCHESTRATOR] plan이 비어있음! intent={intent}")

        # 3. 간단한 인사말(DIRECT)이면 원격 에이전트 호출 없이 즉시 응답
        if intent == "DIRECT" and direct_answer:
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": direct_answer
            }
            return

        # 4. 수립된 Plan에 따라 에이전트들을 순서대로 호출 (바통 터치!)
        if plan: # [ 2 ]
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": f"📋 {len(plan)}개 에이전트에 요청 중..."
            }

            results = []
            all_artifacts = []  # 에이전트 간 데이터 전달용 아티팩트 모음
            previous_result_content = ""  # 이전 에이전트 텍스트 결과를 다음 에이전트에 전달
            previous_step_failed = False  # 이전 단계 실패 여부 (실패 시 후속 단계 중단)
            file_list_from_artifacts = []  # File Agent의 DataPart에서 추출한 파일 목록

            for i, step in enumerate(plan):
                agent_name = step.get("agent")
                agent_query = step.get("query", query)

                # [안전장치] 이전 스텝이 실패했으면 현재 스텝은 스킵합니다.
                if previous_step_failed:
                    logger.warning(f"[ORCHESTRATOR] 이전 스텝 실패로 {agent_name} 스킵")
                    results.append({
                        "agent": agent_name,
                        "success": False,
                        "content": "이전 단계 실패로 스킵됨"
                    })
                    continue

                # [바통 터치 A] File Agent가 찾은 파일 목록(storage_ref 포함)을 RAG Agent에게 전달!
                if agent_name == "internal_rag" and file_list_from_artifacts: # [ 1 ]
                    files_text = "\n".join([
                        f"- {f['filename']} ({f['storage_ref']})"
                        for f in file_list_from_artifacts
                    ])
                    agent_query = f"{agent_query}\n\n사용 가능한 파일 목록:\n{files_text}"
                    logger.info(f"[ORCHESTRATOR] RAG에 {len(file_list_from_artifacts)}개 파일 정보 전달")
                # [바통 터치 B] 이전 에이전트의 텍스트 결과(예: 웹 검색 요약본)를 다음 에이전트(예: 파일 저장)에 전달!
                elif previous_result_content and i > 0:
                    if "[이전 결과]" in agent_query or "[검색 결과]" in agent_query:
                        agent_query = agent_query.replace("[이전 결과]", previous_result_content)
                        agent_query = agent_query.replace("[검색 결과]", previous_result_content)
                    else:
                        agent_query = f"{agent_query}\n\n[이전 에이전트 결과]:\n{previous_result_content[:2000]}"

                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"🔄 {agent_name} 에이전트 호출 중..."
                }

                logger.info(f"[ORCHESTRATOR] [CALL] {agent_name} 호출: {agent_query[:200]}...") # [ 2 ]
                result = await self.call_remote_agent(agent_name, agent_query)
                logger.info(f"[ORCHESTRATOR] [CALL] {agent_name} 결과: success={result['success']}, content={result['content'][:100] if result['content'] else 'None'}")

                actual_success = result["success"]
                previous_step_failed = not actual_success

                # File Agent 호출이 성공했다면 DataPart 아티팩트에서 파일 목록(storage_ref)을 꺼내 보관합니다.
                if agent_name == "file_management" and actual_success: # [ 3 ]
                    for artifact in result.get("artifacts", []):
                        artifact_name = artifact.get("name")
                        artifact_data = artifact.get("data")

                        if artifact_name == "file_list" and isinstance(artifact_data, dict):
                            files = artifact_data.get("files", [])
                            if files:
                                file_list_from_artifacts.extend(files)
                                logger.info(f"[ORCHESTRATOR] DataPart에서 {len(files)}개 파일 추출")

                results.append({
                    "agent": agent_name,
                    "success": actual_success,
                    "content": result["content"]
                })
                all_artifacts.extend(result.get("artifacts", []))

                if actual_success:
                    previous_result_content = result["content"]

            # 5. 모든 단계의 결과를 모아 최종 보고서 작성
            yield { # [ 4 ]
                "is_task_complete": False,
                "require_user_input": False,
                "content": "📝 결과를 정리하고 있습니다..."
            }

            final_response = await self.generate_final_response(query, results)

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": final_response,
                "artifacts": all_artifacts
            }
        else:
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "처리할 작업이 없습니다."
            }
