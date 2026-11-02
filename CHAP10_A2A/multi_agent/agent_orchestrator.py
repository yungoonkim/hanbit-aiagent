# ==============================================================================
# 📘 [CHAP10] 멀티 에이전트 오케스트레이터 (AgentOrchestrator)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - AgentOrchestrator = 종합 프로젝트 총괄 매니저(PM)
#   고객(사용자)이 "1752+2523 계산하고, AI 트렌드도 검색해줘" 같은 복합 업무를 요청하면,
#   총괄 PM(gemini-3.8-flash)이 각 전문가 에이전트의 명함(AgentCard)을 보고:
#   1) 계산 업무는 'langgraph' 에이전트(10001번 포트)에게 외주 요청(send_message)
#   2) 웹 검색 업무는 'mcp' 에이전트(10002번 포트)에게 외주 요청(send_message)
#   3) 두 전문가가 보내온 결과물(Artifact)을 하나로 취합해 고객에게 최종 보고서를 전달합니다!
# ==============================================================================

import asyncio
from uuid import uuid4
import os
import json
import httpx
from dotenv import load_dotenv

from a2a.client import A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    SendMessageRequest,
    TextPart,
)
from a2a.client import A2ACardResolver

from openai import AsyncOpenAI

load_dotenv()
load_dotenv(dotenv_path="../.env")


class AgentOrchestrator:
    def __init__(self, agent_urls: dict): # [ 1 ]
        """
        오케스트레이터(총괄 PM) 초기화

        Args:
            agent_urls: 협력할 전문 에이전트들의 이름과 주소록(URL 매핑)
                예: {'langgraph': 'http://localhost:10001', 'mcp': 'http://localhost:10002'}
        """
        self.agent_urls = agent_urls
        self.agents = {}  # 각 전문 에이전트의 명함(card)과 연락처(A2A 클라이언트) 저장소
        self.httpx_client = None
        self.openai_client = None  # Gemini OpenAI 호환 클라이언트 (두뇌 역할)
        self.conversation_history = []  # 대화 이력 저장

    async def initialize(self): # [ 2 ]
        """
        에이전트 네트워크 초기화 (전문가들과 명함 교환 및 두뇌 모델 준비)

        1. 주소록에 있는 각 에이전트의 AgentCard(명함) 가져오기
        2. 각 에이전트와 대화할 A2A 클라이언트 생성
        3. 작업 분배를 판단할 Gemini(gemini-3.8-flash) 클라이언트 생성
        """
        self.httpx_client = httpx.AsyncClient(timeout=60.0)

        # 주소록을 돌면서 각 전문가 에이전트에게 명함을 받아옵니다.
        for name, url in self.agent_urls.items(): # [ 3 ]
            try:
                card_resolver = A2ACardResolver(
                    httpx_client=self.httpx_client,
                    base_url=url,
                )
                agent_card = await card_resolver.get_agent_card()

                client = A2AClient( # [ 4 ]
                    httpx_client=self.httpx_client,
                    agent_card=agent_card,
                )

                self.agents[name] = {
                    'card': agent_card,
                    'client': client,
                    'url': url,
                }

                print(f"{name} 에이전트: {agent_card.name}")

            except Exception as e:
                print(f"{name} 에이전트 초기화 실패: {str(e)}")

        # Google API 키(또는 OpenAI 호환 키)를 확인하여 Gemini OpenAI 호환 엔드포인트로 연결합니다.
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('OPENAI_API_KEY') # [ 5 ]
        if not api_key:
            raise ValueError("GOOGLE_API_KEY (또는 OPENAI_API_KEY) 환경 변수가 설정되지 않았습니다")

        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    async def _call_agent(self, agent_name: str, task: str) -> str:
        """
        실제로 A2A 프로토콜을 통해 특정 전문 에이전트에게 외주 업무를 맡기고 결과를 받아오는 내부 메서드

        Args:
            agent_name: 호출할 에이전트 이름 ('langgraph' 또는 'mcp')
            task: 에이전트에게 시킬 구체적인 작업 내용

        Returns:
            str: 에이전트가 완료하여 보내준 결과물(Artifact 텍스트)
        """
        if agent_name not in self.agents: # [ 1 ]
            available = ", ".join(self.agents.keys())
            return f"에러: '{agent_name}' 에이전트를 찾을 수 없습니다. 사용 가능한 에이전트: {available}"

        agent = self.agents[agent_name] # [ 2 ]
        client = agent['client']

        try:
            # 1. A2A 프로토콜 표준 메시지 생성
            message = Message( # [ 3 ]
                kind='message',
                role='user',
                parts=[TextPart(kind='text', text=task)],
                message_id=uuid4().hex,
            )

            # 2. 메시지 전송 요청서 포장
            request = SendMessageRequest( # [ 4 ]
                id=uuid4().hex,
                params=MessageSendParams(message=message),
            )

            # 3. A2A 클라이언트로 전문가 에이전트에게 요청 전송 (최대 60초 대기)
            response = await asyncio.wait_for( # [ 5 ]
                client.send_message(request),
                timeout=60.0
            )

            # 4. 전문가 에이전트가 보내온 응답(Task 결과)에서 최종 산출물(artifacts) 추출
            result = response.root.result # [ 6 ]
            history = result.history
            artifacts = result.artifacts

            if artifacts:
                for artifact in artifacts:
                    if artifact.parts:
                        texts = [p.root.text for p in artifact.parts if hasattr(p.root, 'text')]
                        if texts:
                            return "\n".join(texts)

            return "응답을 받지 못했습니다"

        except Exception as e:
            return f"에러: {str(e)}"

    async def process_query(self, user_query: str) -> str:
        """
        사용자 질문을 분석하고 전문가 에이전트들을 조율하여 최종 답변을 만듭니다 (Orchestrator 패턴).

        gemini-3.8-flash 모델이 Function Calling(send_message 도구)을 사용해
        필요한 에이전트를 순차적 또는 복합적으로 호출합니다.

        Args:
            user_query: 사용자 질문

        Returns:
            str: 최종 통합 응답
        """
        print(f"\n💬 사용자 질문: {user_query}")
        print("🤔 Orchestrator가 작업 분석 중...")

        # 현재 연결된 에이전트들의 명함 설명을 목록으로 정리합니다.
        agent_list = [] # [ 1 ]
        for name, agent_info in self.agents.items():
            card = agent_info['card']
            agent_list.append(f"- {name}: {card.description}")

        agents_desc = "\n".join(agent_list)

        # 오케스트레이터(총괄 PM)에게 부여할 시스템 지시문 [ 2 ]
        system_message = f"""
당신은 복잡한 작업을 여러 전문 에이전트에게 라우팅하는 Orchestrator입니다.

사용 가능한 에이전트:
{agents_desc}

에이전트별 전문 분야:
- langgraph: 산술 계산 (덧셈, 뺄셈, 곱셈, 나눗셈 등), 날짜/시간 정보 제공
- mcp: 웹 검색 및 최신 정보 조회 (Tavily API 사용)
  * AI/기술 트렌드, 최신 뉴스, 프로그래밍 정보 등
  * 실시간 인터넷 정보가 필요한 모든 질문

작업 방식:
1. 사용자 질문을 분석하여 필요한 작업들을 식별
2. 각 작업에 적합한 에이전트를 선택
   - 계산 → langgraph
   - 웹 검색/최신 정보 → mcp
3. send_message 함수를 호출하여 작업 위임
4. 에이전트의 응답을 받은 후 다음 작업 진행 또는 최종 답변 생성

중요 규칙:
- 각 작업마다 send_message를 1번씩만 호출
- 에이전트 응답에 "에러" 또는 "시간 초과"가 포함되면 재시도하지 말고 바로 최종 답변 생성
- 에러가 발생한 작업은 "해당 정보를 가져올 수 없습니다"로 처리하고 나머지 결과로 답변
- 모든 필요한 작업이 완료되거나 에러가 발생하면 즉시 최종 답변 생성
- 불필요한 반복 호출 금지

예시:
질문: "2+2 계산하고, 2025년 AI 트렌드도 검색해줘"
→ 1. send_message("langgraph", "2+2를 계산해줘")
→ 2. send_message("mcp", "2025년 AI 트렌드를 검색해줘")
→ 3. 두 결과를 통합하여 최종 답변 (에러 발생 시에도 바로 답변)
"""

        # 대화 이력에 시스템 역할과 사용자 질문 추가
        messages = [ # [ 3 ]
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query},
        ]

        # LLM(gemini-3.8-flash)이 사용할 수 있는 '에이전트 호출 도구(send_message)' 명세서
        send_message_tool = { # [ 4 ]
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "특정 에이전트에게 작업을 전달합니다. 여러 번 호출 가능합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": f"호출할 에이전트 이름. 선택 가능: {', '.join(self.agents.keys())}",
                            "enum": list(self.agents.keys()),
                        },
                        "task": {
                            "type": "string",
                            "description": "에이전트에게 전달할 작업 내용",
                        },
                    },
                    "required": ["agent_name", "task"],
                },
            },
        }

        try:
            # 최대 6번의 에이전트 호출 루프 허용 (무한 루프 방지 안전장치)
            max_iterations = 6 # [ 1 ]
            iteration = 0
            function_calls_made = 0  # 실제로 에이전트를 호출한 횟수

            while iteration < max_iterations:
                iteration += 1
                print(f"\n🔄 반복 {iteration}:")

                # gemini-3.8-flash 모델에게 현재 상황과 도구를 주고 판단을 요청합니다.
                response = await self.openai_client.chat.completions.create( # [ 2 ]
                    model=os.getenv('GEMINI_MODEL', 'gemini-3.8-flash'),
                    messages=messages,
                    tools=[send_message_tool],
                    tool_choice="auto",
                    temperature=0.1,
                )

                message = response.choices[0].message

                # Case 1: LLM이 "전문 에이전트에게 외주를 맡겨야겠다(tool_calls)!"라고 판단한 경우
                if message.tool_calls: # [ 3 ]
                    messages.append(message)

                    # 요청된 모든 에이전트 호출을 순서대로 처리합니다.
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == 'send_message':
                            function_calls_made += 1

                            # 누구에게(agent_name) 무슨 일(task)을 시킬지 인자 추출
                            args = json.loads(tool_call.function.arguments)
                            agent_name = args.get('agent_name', '')
                            task = args.get('task', '')

                            print(f"  🎯 {agent_name} 에이전트 호출 중... (#{function_calls_made})")

                            # 실제 A2A 프로토콜로 전문 에이전트 호출!
                            result = await self._call_agent(agent_name, task)

                            # 전문 에이전트가 보내온 결과를 대화 기록에 추가하여 LLM에게 보고합니다.
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            })

                            print(f"  ✅ 결과: {result[:100]}...")

                    # 도구 실행 결과를 들고 다음 반복으로 이동 (추가 호출할지 최종 답변할지 결정)
                    continue

                # Case 2: 모든 외주 작업이 끝나고 LLM이 최종 통합 답변을 작성한 경우
                elif message.content: # [ 4 ]
                    return message.content

                # Case 3: 함수 호출도 없고 내용도 없는 비정상 상황 방어
                else: # [ 5 ]
                    return "응답을 생성할 수 없습니다"

            return "작업이 너무 복잡하여 완료할 수 없습니다. 질문을 단순화해주세요."

        except Exception as e:
            return f"Orchestrator 오류: {str(e)}"

    async def close(self):
        """HTTP 클라이언트 리소스 정리 (퇴근 시 문 잠그기)"""
        if self.httpx_client:
            await self.httpx_client.aclose()


async def main():
    # 협력할 두 전문 에이전트의 서버 주소 등록
    agent_urls = { # [ 1 ]
        'langgraph': 'http://localhost:10001',
        'mcp': 'http://localhost:10002',
    }

    orchestrator = AgentOrchestrator(agent_urls)

    try:
        # 1. 에이전트 명함 교환 및 초기화
        await orchestrator.initialize() # [ 2 ]

        # 2. 단일 질문부터 복합 질문까지 다양한 시나리오 테스트
        test_queries = [ # [ 3 ]
            "안녕하세요!",
            "현재 시각은?",
            "2025년 AI 에이전트 트렌드는?",
            "1752 + 2523를 계산하고, 2025년 AI 트렌드도 검색해주세요",
            "현재 시각과 함께, 파이썬 최신 버전도 알려주세요",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n[테스트 {i}/{len(test_queries)}]")

            response = await orchestrator.process_query(query)

            print(f"\n최종 통합 응답: {response}")

            if i < len(test_queries):
                await asyncio.sleep(3)

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await orchestrator.close()


if __name__ == '__main__':
    asyncio.run(main())
