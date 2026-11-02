# ==============================================================================
# 📘 [CHAP10] 순수 MCP 웹 검색 에이전트 본체 (MCPAgent)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - MCP(Model Context Protocol)가 에이전트에게 '외부 도구(USB 장치, 여기서는 Tavily 검색기)'를
#   꽂아주는 표준 규격이라면, A2A는 그 도구를 가진 에이전트가 '다른 에이전트와 협업'하는 규격입니다!
# - 이 파일은 LangChain 없이 순수 MCP SDK(`streamablehttp_client`, `ClientSession`)와
#   Gemini 모델(`gemini-3.8-flash`)을 직접 연결하여 실시간 웹 검색을 수행합니다.
# ==============================================================================

from typing import AsyncIterator, Dict, Any, List, Optional
from dotenv import load_dotenv
import os
import json
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from contextlib import asynccontextmanager

load_dotenv()
load_dotenv(dotenv_path="../../.env")


class MCPAgent: # [ 1 ]
    """
    Tavily 원격 MCP 서버와 연결되어 실시간 인터넷 검색을 수행하는 에이전트
    """
    def __init__(self):
        """에이전트 초기화"""
        self.mcp_url = None
        self.tools = []
        self.initialized = False
        self.openai_client = None
        self._current_session: Optional[ClientSession] = None
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")

    @asynccontextmanager
    async def _get_mcp_session(self): # [ 2 ]
        """
        MCP 서버와의 통신 세션을 안전하게 열고 닫는 Context Manager
        비유: 도서관(Tavily MCP 서버)에 전화선을 연결하고 대화가 끝나면 자동으로 수화기를 내려놓는 장치
        """
        async with streamablehttp_client(self.mcp_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def initialize(self): # [ 3 ]
        """
        Tavily MCP 서버에 접속하여 사용할 수 있는 검색 도구 목록(list_tools)을 가져오고,
        두뇌 역할을 할 Gemini OpenAI 호환 클라이언트를 초기화합니다.
        """
        if self.initialized:
            return

        try:
            self.mcp_url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={self.tavily_api_key}"
            self.openai_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

            # MCP 서버에 접속해 제공되는 도구(예: tavily_search 등) 목록을 받아옵니다.
            async with self._get_mcp_session() as session:
                tools_result = await session.list_tools()
                self.tools = tools_result.tools

                for tool in self.tools:
                    print(f"  - {tool.name}: {tool.description or ''}")

            self.initialized = True

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str: # [ 1 ]
        """
        MCP 서버에 있는 도구를 실제로 호출(call_tool)하고 검색 결과를 텍스트로 받아옵니다.

        Args:
            tool_name: 실행할 MCP 도구 이름 (예: 'tavily-search')
            parameters: 도구에 전달할 검색어 등 파라미터

        Returns:
            str: 도구 실행 결과 텍스트
        """
        try:
            # 새 MCP 세션으로 도구 실행 요청
            async with self._get_mcp_session() as session: # [ 2 ]
                result = await session.call_tool(tool_name, arguments=parameters)

                # 결과에서 텍스트 추출
                if result.content: # [ 3 ]
                    for item in result.content:
                        if hasattr(item, 'text'):
                            return item.text
                    # text 속성이 없으면 전체 내용 반환
                    return str(result.content[0])

                return "도구 실행 완료 (결과 없음)"

        except Exception as e:
            return f"도구 실행 중 오류 발생: {str(e)}"


    async def process_query(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        """
        사용자 질문을 처리합니다.
        gemini-3.8-flash 모델을 사용하여 질문 의도를 파악하고, 필요시 MCP 검색 도구를 호출한 뒤
        검색 결과를 종합하여 최종 답변을 생성합니다.

        Args:
            query: 사용자 질문

        Yields:
            Dict: 진행 상태 및 에이전트 최종 응답
        """
        if not self.initialized:
            await self.initialize()

        try:
            # 1. MCP 도구 스키마를 Function Calling 형식으로 변환합니다.
            tools_for_openai = [] # [ 1 ]
            for tool in self.tools:
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
                tools_for_openai.append(tool_schema)

            # 2. LLM에게 전달할 메시지 구성
            messages = [ # [ 2 ]
                {
                    "role": "system",
                    "content": "당신은 웹 검색을 수행할 수 있는 유용한 어시스턴트입니다. 사용자의 질문에 답하기 위해 필요하다면 tavily_search 도구를 사용하세요."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "질문을 분석하고 있습니다...",
            }

            # 3. gemini-3.8-flash 모델에게 질문과 도구 목록을 전달하여 첫 번째 판단 요청
            response = await self.openai_client.chat.completions.create( # [ 3 ]
                model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                messages=messages,
                tools=tools_for_openai,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # 4. 모델이 MCP 검색 도구를 호출하기로 결정한 경우
            if response_message.tool_calls: # [ 1 ]
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"도구 실행: {tool_name}\n인자: {json.dumps(tool_args, ensure_ascii=False, indent=2)}",
                    }

                    # 실제 MCP 서버에 도구 실행 요청!
                    tool_result = await self.execute_tool(tool_name, tool_args)

                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"도구 실행 결과:\n{tool_result[:500]}{'...' if len(tool_result) > 500 else ''}",
                    }

                    # 도구 호출 내역과 검색 결과를 대화 맥락(messages)에 추가합니다.
                    messages.append({ # [ 2 ]
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # 5. 검색 결과를 바탕으로 gemini-3.8-flash가 최종 요약 답변을 작성합니다.
                final_response = await self.openai_client.chat.completions.create( # [ 3 ]
                    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                    messages=messages
                )

                final_content = final_response.choices[0].message.content

                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": f"최종 답변:\n{final_content}",
                }
            else: # [ 4 ]
                # 도구 호출 없이 바로 답변한 경우
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": response_message.content,
                }

        except Exception as e:
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": f"에러 발생: {str(e)}",
            }

    async def stream(
        self,
        query: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        에이전트를 실행하고 결과를 스트리밍합니다.

        Args:
            query: 사용자 질문

        Yields:
            Dict: 에이전트 응답
        """
        async for response in self.process_query(query):
            yield response
