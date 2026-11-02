# ============================================================================
# [7.6 기획(Planning) 기반 슈퍼바이저] 현재 단계 실행을 관리하는 슈퍼바이저 (supervisor_agent.py)
# 비유: 수석 기획자(planning)가 건네준 체크리스트의 '맨 첫 번째 할 일(plan[0])'만 딱 집중해서
#       리서처(research)나 작가(canvas)에게 지시하고, 그 할 일이 끝나면 완료 도장을 찍어(past_steps)
#       다시 기획자에게 체크리스트 업데이트를 요청하는 '현장 매니저(Supervisor)'입니다.
# ============================================================================
from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from settings import get_model, SupervisorState

# [ 1 ] 현재 작업을 수행할 담당자(canvas 또는 research)를 선택하는 라우터 스키마입니다.
class Router(TypedDict): # [ 1 ]
    """작업을 수행할 에이전트를 라우팅합니다."""
    next: Literal["canvas", "research"]

def supervisor_node(state: SupervisorState) -> Command[Literal["canvas", "research", "planning"]]:
    """Supervisor 노드: 현재 단계(plan[0])의 작업 할당 및 완료 기록을 통합 처리합니다."""

    # [ 2 ] 현재 상태의 계획 리스트, 완료된 단계 기록, 메시지 목록을 가져옵니다.
    plan = state.get("plan", []) # [ 2 ]
    past_steps = state.get("past_steps", [])
    messages = state["messages"]

    # [ 3 ] 계획 리스트의 첫 번째 작업(plan[0])을 현재 수행할 목표로 꺼냅니다.
    try:
        task_description = plan[0]
    except IndexError:
        return Command(goto="planning", update={})

    # 최근 3개의 메시지를 확인하여 방금 하위 에이전트가 현재 작업을 끝냈는지 파악합니다.
    recent_messages = "\n".join([
        f"{msg.name if hasattr(msg, 'name') else 'user'}: {msg.content}"
        for msg in messages[-3:] if messages
    ])

    print("recent_messages:", recent_messages)

    system_prompt = f"""
    당신은 멀티 에이전트 시스템의 수퍼바이저입니다. **오직 현재 작업만 관리**하세요.
    Router 도구를 사용하여 작업을 수행할 에이전트를 결정하세요.

    ## 현재 작업 (이것만 처리하세요):
    {task_description}

    ## 팀 멤버:
    - canvas: 글 작성, 목차 작성, 파일 저장을 담당
    - research: 웹 검색기반 분석, 자료 조사 등을 담당

    ## 최근 대화 내역:
    {recent_messages}
    **research_agent 혹은 canvas_agent가 이미 현재 작업 "{task_description}"을 완료했는지 판단**하고, 완료된 경우 또 다시 에이전트를 호출하지 말고 planning으로 돌아가세요.
    """

    llm = get_model("gemini-3.8-flash")

    # [ 4 ] Router 도구를 바인딩하여 현재 작업을 누구에게 맡길지(혹은 이미 완료되었는지) 판단합니다.
    response = llm.bind_tools([Router]).invoke([{"role": "system", "content": system_prompt}]) # [ 4 ]

    # 아직 현재 작업이 수행되지 않아 Router 도구가 호출된 경우 -> 해당 에이전트(research/canvas)로 이동!
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        router_args = response.tool_calls[0]["args"]
        goto = router_args["next"]

        task_message = HumanMessage(content=f"작업을 수행해주세요: {task_description}")

        return Command(
            goto=goto,
            update={
                "next": goto,
                "messages": [task_message]
            }
        )

    # [ 5 ] 이미 하위 에이전트가 현재 작업을 완료한 경우 -> past_steps에 결과를 기록하고 기획자(planning)에게 복귀!
    else: # [ 5 ]
        for msg in reversed(messages):
            if hasattr(msg, 'name') and msg.name in ["canvas_agent", "research_agent"]:
                supervisor_final_message = f"작업 '{task_description}'이 완료되었습니다.\n\n작업 결과:\n{msg.content}"
                break

        updated_past_steps = past_steps + [(task_description, supervisor_final_message)]

        completion_message = AIMessage(
            content=supervisor_final_message,
            name="supervisor"
        )

        return Command(
            goto="planning",
            update={
                "past_steps": updated_past_steps,
                "messages": [completion_message]
            }
        )
