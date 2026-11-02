# ==============================================================================
# 📘 [CHAP11] 웹 리서치 에이전트 본체 (WebResearchAgent)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - 이 에이전트는 외부 인터넷 세상의 최신 뉴스와 기술 동향을 수집해오는 '해외/외부 특파원'입니다!
# - `MultiServerMCPClient`를 사용해 Tavily 원격 MCP 서버의 검색 도구를 자동으로 불러오고,
#   `ChatGoogleGenerativeAI(model="gemini-3.8-flash")` 두뇌와 결합하여 최신 정보를 조사·요약합니다.
# ==============================================================================

import os
import sys
import logging
import asyncio
from typing import AsyncIterator, Dict, Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

logger = logging.getLogger(__name__)

# Windows 이벤트 루프 정책 설정
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # [ 1 ]

SYSTEM_PROMPT = """당신은 웹 검색 전문가입니다.

tavily_search 도구 사용 시 주의사항:
- topic 파라미터는 반드시 'general', 'news', 'finance' 중 하나만 사용하세요.
- 기술, AI, 프로그래밍 관련 질문도 topic='general'을 사용하세요.
- 최신 뉴스 검색 시에만 topic='news'를 사용하세요.
- 금융/주식 관련 검색 시에만 topic='finance'를 사용하세요.
"""


class WebResearchAgent:
    """
    MCP 기반 웹 검색 에이전트 (외부 정보 리서치 특파원)

    Tavily MCP 서버를 통해 실시간 웹 검색을 수행하고 gemini-3.8-flash로 요약합니다.
    """

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
        self.mcp_client = None
        self.agent = None
        self.initialized = False

    async def initialize(self) -> None:
        """Tavily MCP 클라이언트 연결 및 LangChain 에이전트 초기화"""
        if self.initialized:
            return

        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY 환경 변수가 필요합니다")

        # LangChain MCP 어댑터를 사용해 원격 Tavily MCP 서버에 연결합니다.
        self.mcp_client = MultiServerMCPClient({ # [ 2 ]
            "tavily": {
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
                "transport": "streamable_http",
            }
        })

        # MCP 서버가 제공하는 웹 검색 도구들을 LangChain 도구 형식으로 가져옵니다.
        tools = await self.mcp_client.get_tools(server_name="tavily")
        logger.info(f"[WEB AGENT] [INIT] Tavily 도구 로드: {[t.name for t in tools]}")

        self.agent = create_agent(self.model, tools)
        self.initialized = True
        logger.info("[WEB AGENT] [INIT] 초기화 완료")

    async def stream( # [ 3 ]
        self,
        query: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """질문을 받아 웹 검색을 수행하고 진행 상태 및 최종 요약 결과를 스트리밍합니다."""
        if not self.initialized:
            await self.initialize()

        try:
            final_message = None
            tool_called = False

            async for chunk in self.agent.astream({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ]
            }):
                for node_name, node_output in chunk.items():
                    messages = node_output.get("messages", [])

                    for msg in messages:
                        # 검색 도구 호출 시 진행 상태 알림
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            if not tool_called:
                                tool_called = True
                                tool_names = [tc.get('name', 'unknown') for tc in msg.tool_calls]
                                yield {
                                    "is_task_complete": False,
                                    "require_user_input": False,
                                    "content": f"🔍 웹 검색 중... (도구: {', '.join(tool_names)})"
                                }

                        # 최종 요약 답변 추출
                        if hasattr(msg, 'content') and msg.content:
                            if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                                final_message = msg.content

            if final_message:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": final_message
                }
            else:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "응답을 생성하지 못했습니다."
                }

        except Exception as e:
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"처리 중 오류 발생: {str(e)}"
            }


if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("Web Research Agent 테스트")
        print("=" * 60)

        agent = WebResearchAgent()
        query = "2026년 AI 에이전트 트렌드를 조사해주세요."

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
