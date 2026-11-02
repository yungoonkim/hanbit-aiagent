# ==============================================================================
# 📘 [CHAP10] LangGraph 에이전트 본체 (LangGraphAgent)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - 이 에이전트는 '계산기(calculator)'와 '시계(get_current_info)' 두 가지 도구를 가진
#   **수학 및 시간 전문 비서**입니다.
# - ResponseFormat = 작업이 다 끝났는지(completed), 추가 질문이 필요한지(input_required)를
#   정해진 양식(체크리스트)으로 보고하도록 강제하는 '업무 완료 보고서 양식'입니다.
# ==============================================================================

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from typing import AsyncIterator, Dict, Any, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path="../../.env")


class ResponseFormat(BaseModel): # [ 1 ]
    """
    LLM이 최종 응답할 때 반드시 따라야 할 구조화된 응답 포맷 (업무 보고서 양식)

    - status: 작업 완료('completed'), 사용자 추가 입력 필요('input_required'), 에러('error') 중 하나
    - message: 사용자(또는 오케스트레이터)에게 전달할 최종 답변 내용
    """
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


@tool
def calculator(expression: str) -> str: # [ 2 ]
    """수학 계산을 수행하는 도구입니다 (예: 덧셈, 뺄셈, 곱셈, 나눗셈).

    Args:
        expression (str): 계산할 수학 표현식 (예: "2 + 2", "1752 + 2523")

    Returns:
        str: 계산 결과 또는 오류 메시지
    """
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def get_current_info() -> str: # [ 3 ]
    """현재 날짜와 시간 정보를 조회하는 도구입니다.

    Returns:
        str: 포맷팅된 현재 시각 문자열
    """
    from datetime import datetime
    now = datetime.now()
    return f"현재 시각: {now.strftime('%Y년 %m월 %d일 %H시 %M분')}"


class LangGraphAgent: # [ 4 ]
    """
    LangGraph 기반의 수학 계산 및 시간 조회 전문 에이전트
    """
    SYSTEM_INSTRUCTION = (
        "당신은 도움이 되는 AI 어시스턴트입니다. "
        "사용자의 질문에 답변하고, 필요한 경우 도구를 사용하세요."
    )

    def __init__(self):
        """에이전트 초기화: gemini-3.8-flash 모델, 도구, 메모리 체크포인터를 결합합니다."""
        # 1. 두뇌 역할을 할 Gemini 모델(gemini-3.8-flash) 초기화
        self.llm = ChatGoogleGenerativeAI( # [ 5 ]
            model='gemini-3.8-flash'
        )

        # 2. 사용할 도구 가방 준비 (계산기, 시계)
        self.tools = [calculator, get_current_info]

        # 3. 대화 맥락을 기억할 인메모리 저장소(체크포인터) 준비
        self.memory = InMemorySaver()

        # 4. 모델 + 도구 + 프롬프트 + 메모리 + 응답 양식을 하나로 묶어 에이전트 생성
        self.agent = create_agent( # [ 6 ]
            self.llm,
            tools=self.tools,
            system_prompt=self.SYSTEM_INSTRUCTION,
            checkpointer=self.memory,
            response_format=ResponseFormat,
        )

    async def stream( # [ 1 ]
        self,
        query: str,
        session_id: str = "default"
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        질문을 받아 에이전트를 실행하고 진행 상황을 실시간(스트리밍)으로 알려줍니다.

        비유: 배달 앱에서 '조리 시작(도구 실행 중) -> 조리 완료(도구 결과) -> 배달 완료(최종 답변)'
        상태를 실시간으로 중계해주는 것과 같습니다!
        """
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': session_id}}

        try: # [ 2 ]
            # 에이전트 그래프가 한 단계씩 진행될 때마다 상태를 받아옵니다.
            async for item in self.agent.astream(inputs, config, stream_mode='values'):
                if 'messages' in item:
                    message = item['messages'][-1]

                    # AI가 도구(계산기/시계)를 쓰기로 결정했을 때 중간 상태 알림
                    if ( # [ 3 ]
                        isinstance(message, AIMessage)
                        and message.tool_calls
                        and len(message.tool_calls) > 0
                    ):
                        yield {
                            'is_task_complete': False,
                            'require_user_input': False,
                            'content': '도구를 실행하는 중...',
                        }

                    # 도구 실행 결과가 나왔을 때 중간 상태 알림
                    elif isinstance(message, ToolMessage): # [ 4 ]
                        yield {
                            'is_task_complete': False,
                            'require_user_input': False,
                            'content': message.content,
                        }

            # 모든 단계가 끝나면 최종 구조화된 응답(ResponseFormat)을 확인하여 반환합니다.
            yield self.get_agent_response(config) # [ 5 ]

        except Exception as e:
            yield {
                'is_task_complete': False,
                'require_user_input': False,
                'content': f"에러 발생: {str(e)}",
            }

    def get_agent_response(self, config):
        """
        그래프의 최종 상태(State)에서 ResponseFormat 양식에 맞춘 결과를 꺼내어
        A2A 실행자(AgentExecutor)가 이해하기 쉬운 딕셔너리로 변환합니다.
        """
        current_state = self.agent.get_state(config) # [ 1 ]
        structured_response = current_state.values.get('structured_response')

        if structured_response and isinstance(structured_response, ResponseFormat): # [ 2 ]
            # 추가 입력이 필요하거나 에러인 경우
            if structured_response.status in ('input_required', 'error'):
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            # 작업이 정상적으로 완료된 경우
            if structured_response.status == 'completed': # [ 3 ]
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                '요청을 처리할 수 없습니다. '
                '다시 시도해주세요.'
            ),
        }
