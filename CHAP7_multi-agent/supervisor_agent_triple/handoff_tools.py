# ============================================================================
# [7.5 핸드오프 도구 기반 3중 슈퍼바이저] 핸드오프(바통 터치) 도구 생성기 (handoff_tools.py)
# 비유: 팀장님이 각 부서(web_search, db_search, faq)에 업무를 지시할 때 누르는 '내선 전화 버튼'입니다.
#       - 한 부서에만 일을 시킬 때는 일반 Command(goto=부서명)로 연결하고,
#       - 두 곳 이상의 부서에 동시에 일을 시킬 때는 Send()를 이용해 병렬로 업무를 배정합니다!
# ============================================================================
import uuid
from typing import cast
from langchain.tools import tool, ToolRuntime
from langchain.messages import AIMessage, ToolMessage
from langgraph.types import Command, Send


def create_handoff_tool(agent_name: str, description: str):
    """
    supervisor 에이전트에서 다른 에이전트로 작업을 handoff(위임)하는 도구를 동적으로 생성합니다.
    """

    # [ 1 ] 도구 이름을 'transfer_to_에이전트이름' 형태로 자동 지정합니다.
    tool_name = f"transfer_to_{agent_name}" # [ 1 ]

    @tool(tool_name, description=description)
    def handoff_to_agent(
        query: str,
        runtime: ToolRuntime, # [ 2 ] LangGraph 런타임이 현재 그래프 상태(state)와 tool_call_id를 자동으로 주입합니다.
    ) -> Command:
        """
        에이전트로 작업을 handoff하는 도구입니다.

        Args:
            query: 에이전트에게 전달할 쿼리 (최대한 원본 쿼리를 유지)
            runtime: 도구 런타임 (런타임에서 자동 주입)
        """
        # [ 3 ] 현재 그래프 상태와 마지막 AI 메시지(도구 호출 정보가 담긴 메시지)를 가져옵니다.
        state = runtime.state # [ 3 ]
        tool_call_id = runtime.tool_call_id
        last_ai_message = cast(AIMessage, state["messages"][-1])

        # 2개 이상의 도구 호출(복수 부서 동시 지시): 모든 handoff 도구를 한번에 Send로 병렬 처리합니다.
        if len(last_ai_message.tool_calls) > 1:
            handoff_messages = state["messages"][:-1]

            # 모든 handoff 도구 호출을 수집할 리스트
            send_list = []
            # [ 4 ] 슈퍼바이저가 호출한 여러 개의 도구들을 하나씩 순회합니다.
            for tool_call in last_ai_message.tool_calls: # [ 4 ]
                # transfer_to_ 로 시작하는 handoff 도구만 처리
                if tool_call["name"].startswith("transfer_to_"):
                    target_agent = tool_call["name"].replace("transfer_to_", "")
                    query_content = tool_call['args'].get('query', '')

                    filtered_ai_message = AIMessage(
                        content=last_ai_message.content,
                        tool_calls=[tool_call],
                        name=last_ai_message.name,
                        id=str(uuid.uuid4()),
                    )

                    tool_msg = ToolMessage(
                        content=f"성공적으로 {target_agent}에게 작업을 전달했습니다: {query_content}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )

                    # [ 5 ] 각 하위 에이전트에게 전달할 메시지와 전용 쿼리(query)를 Send 객체로 묶습니다.
                    agent_messages = handoff_messages + [filtered_ai_message, tool_msg] # [ 5 ]
                    send_list.append(Send(target_agent, {
                        **state,
                        "messages": agent_messages,
                        "query": query_content
                    }))

            # 모든 대상 에이전트에게 한번에 병렬로 Send 명령을 내립니다. (부모 그래프 기준 이동)
            return Command(
                graph=Command.PARENT,
                goto=send_list,
            )

        # [ 6 ] 1개 도구 호출(단일 부서 지시): 해당 에이전트로 바로 handoff 처리합니다.
        else: # [ 6 ]
            tool_message = ToolMessage(
                content=f"성공적으로 {agent_name}에게 작업을 전달했습니다: {query}",
                name=tool_name,
                tool_call_id=tool_call_id,
            )

            handoff_messages = state["messages"] + [tool_message]
            return Command(
                goto=agent_name,
                graph=Command.PARENT,
                update={**state, "messages": handoff_messages, "query": query}
            )

    return handoff_to_agent


def create_handoff_messages(agent_name: str):
    """
    하위 에이전트가 작업을 마치고 supervisor(팀장님)에게 돌아갈 때 남기는 완료 보고 메시지 쌍을 생성합니다.

    Args:
        agent_name: 현재 에이전트 이름

    Returns:
        tuple: (AIMessage, ToolMessage) 쌍
    """
    tool_call_id = str(uuid.uuid4())
    tool_name = f"transfer_back_to_supervisor"

    # 슈퍼바이저가 프롬프트 규칙에서 이 문구("Supervisor로 이동합니다.")를 보고 직전 메시지가 결과물임을 인식합니다.
    ai_message = AIMessage(
        content=f"Supervisor로 이동합니다.",
        name=agent_name,
        tool_calls=[{
            "name": tool_name,
            "args": {},
            "id": tool_call_id
        }]
    )

    tool_message = ToolMessage(
        content=f"supervisor로 성공적으로 작업을 전달했습니다.",
        name=tool_name,
        tool_call_id=tool_call_id,
    )

    return ai_message, tool_message
