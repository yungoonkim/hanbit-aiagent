# ============================================================================
# [7.4 슈퍼바이저 패턴 기초] 팀원들을 지휘하는 슈퍼바이저 노드 (supervisor_agent.py)
# 비유: 고객의 요청을 접수한 '팀장님'입니다.
#       1) 아직 할 일이 남았으면 Router 도구를 사용해 web_agent나 database_agent에게 지시를 내리고,
#       2) 모든 팀원의 보고가 끝났으면 결과를 종합해 고객에게 최종 답변을 전달하고 퇴근(END)합니다.
# ============================================================================
from typing import Literal, TypedDict
from langchain.messages import AIMessage
from langgraph.types import Command
from langgraph.graph import MessagesState, END

from settings import get_model, get_system_prompt

# [ 1 ] 팀장님이 다음 작업자를 지목할 때 사용할 라우팅 스키마(도구)입니다.
class Router(TypedDict): # [ 1 ]
    """다음으로 라우팅 할 작업자"""

    next: Literal["web_agent", "database_agent"]

class State(MessagesState):
    next: str

# Gemini 3.8 Flash 모델을 불러옵니다.
llm = get_model("gemini-3.8-flash")

def supervisor_node(state: State) -> Command[Literal["web_agent", "database_agent", END]]:
    """대화의 맥락을 파악해 적절한 하위 에이전트로 라우팅하거나 최종 답변을 생성하는 노드입니다."""
    # [ 2 ] 슈퍼바이저 시스템 프롬프트와 현재까지의 대화 기록을 합칩니다.
    messages = [ # [ 2 ]
        {"role": "system", "content": get_system_prompt(["web_agent", "database_agent"])},
    ] + state["messages"]

    # Router 도구를 바인딩하여 LLM에게 "누구에게 일을 시킬지, 아니면 직접 마무리할지" 판단하게 합니다.
    response = llm.bind_tools([Router]).invoke(messages)

    # [ 3 ] 만약 LLM이 Router 도구를 호출했다면 -> 지목된 팀원(goto)에게 작업을 넘깁니다!
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0: # [ 3 ]
        goto = response.tool_calls[0]["args"]["next"]
        return Command(goto=goto, update={"next": goto})

    # [ 4 ] 도구 호출 없이 일반 텍스트로 답했다면 -> 모든 작업이 끝난 것이므로 최종 답변을 남기고 종료(END)합니다!
    else: # [ 4 ]
        final_message = AIMessage(content=response.content, name="supervisor")
        return Command(
            goto=END,
            update={"messages": [final_message]}
        )
