# ============================================================================
# [6.2절] 웹 검색 에이전트 만들기 (web_agent/agent.py)
# ============================================================================
# 이 파일은 AI가 스스로 판단하여 인터넷 검색(TavilySearch)을 수행하고,
# 그 검색 결과를 바탕으로 정확한 답변을 만들어내는 '웹 검색 에이전트'입니다.
#
# 💡 비전공자를 위한 일상 비유:
# - 일반 LLM(챗봇)이 "자기 머릿속 지식만으로 시험을 치르는 학생"이라면,
# - 웹 검색 에이전트는 "모르는 문제가 나오면 스마트폰(검색 도구)으로 찾아보고 답을 적는 오픈북 시험 학생"입니다!
# ============================================================================

from typing import Annotated
from typing_extensions import TypedDict

# 랭그래프(LangGraph)에서 그래프 설계도(StateGraph)와 출발점(START), 도착점(END)을 가져옵니다.
from langgraph.graph import StateGraph, START, END
# 대화 기록을 덮어쓰지 않고 카카오톡 대화방처럼 차곡차곡 쌓아주는 리듀서(add_messages)입니다.
from langgraph.graph.message import add_messages

# Google의 최신 AI 모델(Gemini)과 AI 전용 웹 검색 도구(TavilySearch)를 가져옵니다.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# .env 파일에 보관된 비밀 API 키(GOOGLE_API_KEY, TAVILY_API_KEY)를 불러오는 도구입니다.
from dotenv import load_dotenv

# 내 컴퓨터의 .env 파일을 읽어 환경 변수로 등록합니다. (비밀번호 금고 열기!)
load_dotenv()

### 도구 정의 ###
# 비유: AI 학생의 책상 위에 올려둘 '스마트폰 검색 앱(도구)'을 준비하는 단계입니다.

# TavilySearch: AI가 읽기 좋게 광고를 빼고 핵심 본문만 검색해 주는 검색 엔진입니다.
# max_results=3 은 "검색 결과 상위 3개까지만 가져와!"라는 뜻입니다.
tool = TavilySearch(max_results=3)
tools = [tool]  # AI에게 쥐여줄 도구들을 리스트(가방)에 담습니다.

# 똑똑한 두뇌 역할을 할 Gemini 모델을 준비합니다.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# bind_tools(tools): AI 모델에게 "너에게는 이런 검색 도구가 있으니 필요할 때 써!"라고 알려줍니다.
# 주의: 도구를 실제로 실행하는 것이 아니라, AI에게 '도구 사용 설명서'를 쥐여주는 과정입니다.
llm_with_tools = llm.bind_tools(tools)


### 그래프 상태 정의 ###
# 비유: 작업자(노드)들이 돌려보며 대화 내용을 적어두는 '공유 업무 메모장(State)'입니다.

class State(TypedDict):
    # messages 칸에는 대화 목록(list)을 저장하되, 새 메시지가 오면 지우지 않고 뒤에 이어붙입니다(add_messages).
    messages: Annotated[list, add_messages]

# 방금 만든 State 양식을 사용하는 빈 그래프 설계도(graph_builder)를 펼칩니다.
graph_builder = StateGraph(State)


### 그래프 노드 추가 ###
# 비유: 각 부서에서 실제로 일을 처리할 담당 직원(노드)들을 배치합니다.

#### 챗봇 노드 ####
# 역할: 사용자의 질문을 보고 1) 바로 대답할지, 2) 검색 도구를 써달라고 요청(tool_calls)할지 결정하는 두뇌 부서입니다.
def chatbot(state: State):
    # 지금까지 쌓인 대화 기록(state["messages"])을 도구 설명서가 연결된 AI에게 전달합니다.
    response = llm_with_tools.invoke(state["messages"])
    # AI의 응답(일반 답변 또는 도구 호출 요청서)을 메모장의 messages에 추가하기 위해 반환합니다.
    return {"messages": [response]}

# 설계도에 "chatbot"이라는 이름으로 챗봇 노드를 등록합니다.
graph_builder.add_node("chatbot", chatbot)

#### 도구 실행 노드 ####
# 비유: AI 두뇌(chatbot)가 "이 키워드로 검색 좀 해줘!"라고 주문서를 내밀면,
# 실제로 인터넷 검색을 실행해서 결과물을 가져다주는 '심부름꾼(손발) 부서'입니다.

import json
from langchain_core.messages import ToolMessage

class BasicToolNode:
    """
        마지막 AIMessage에서 요청된 도구를 실행하는 노드
    """

    def __init__(self, tools: list) -> None:
        # 도구 이름을 키(Key)로, 실제 도구 객체를 값(Value)으로 하는 사전(Dictionary)을 만듭니다.
        # 비유: 심부름꾼이 "tavily_search 가져와!"라는 말을 들었을 때 바로 꺼낼 수 있게 이름표를 붙여 정리해 둔 공구함입니다.
        self.tools_by_name = {tool.name: tool for tool in tools} # ["tavily_search" : TavilySearch()]

    def __call__(self, inputs: dict):
        # 메모장(inputs)에서 대화 목록(messages)을 꺼내 가장 마지막 메시지(방금 AI가 한 말)를 확인합니다.
        if messages := inputs.get("messages", []): # [ 1 ]
            message = messages[-1]
        else:
            raise ValueError("ERROR: 입력에 메시지가 없습니다.")

        outputs = []
        # AI가 요청한 도구 주문서 목록(message.tool_calls)을 하나씩 꺼내어 처리합니다.
        for tool_call in message.tool_calls: # 메시지에서 호출된 도구를 불러옴
            # 공구함에서 해당 이름의 도구를 찾아, AI가 적어준 검색어(args)를 넣어 실제로 실행(invoke)합니다.
            tool_result = self.tools_by_name[tool_call["name"]].invoke( # [ 2 ] Tool 호출 실행
                tool_call["args"]
            )
            # 검색 결과를 AI가 읽을 수 있는 전용 봉투(ToolMessage)에 담아 결과 리스트에 추가합니다.
            # ensure_ascii=False를 설정해야 한글이 깨지지 않고 예쁘게 들어갑니다!
            outputs.append( # [ 3 ] Tool 호출 결과(ToolMessage) 추가
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],  # 어떤 주문서에 대한 결과인지 고유 번호(ID)를 꼭 맞춰줍니다.
                )
            )
        # 실행 결과 메시지들을 메모장(messages)에 추가합니다.
        return {"messages": outputs}

# 심부름꾼 노드 객체를 만들고 설계도에 "tools"라는 이름으로 등록합니다.
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


### 조건부 엣지 추가 ###
# 비유: 챗봇 부서가 일을 마친 뒤, 다음으로 '심부름꾼 부서(tools)'로 갈지 '퇴근(END)'할지 안내하는 교통정리 요원입니다.

def route_tools(
    state: State,
):
    """
    마지막 메시지에 도구 호출이 있는 경우, ToolNode로 라우팅하고 그렇지 않으면 END로 라우팅
    """
    # 현재 상태에서 가장 마지막 AI 메시지를 꺼냅니다.
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"ERROR: 입력에 메시지가 없습니다. 상태: {state}")

    # 만약 AI 메시지에 '도구 호출 주문서(tool_calls)'가 1개라도 들어있다면 -> "tools" 노드로 이동!
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    # 도구 호출 없이 일반 텍스트 답변을 마쳤다면 -> 그래프 종료(END)!
    return END


# 챗봇 노드 뒤에 갈림길(조건부 엣지)을 연결합니다.
# route_tools가 "tools"를 반환하면 "tools" 노드로, END를 반환하면 종료(END)로 이동합니다.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)


### 나머지 엣지 추가 및 그래프 컴파일 ###

# 심부름꾼("tools")이 검색 결과를 가져오면, 다시 두뇌("chatbot")에게 돌아가서 최종 답변을 작성하게 합니다! (순환 고리)
graph_builder.add_edge("tools", "chatbot")
# 출발점(START)에서 시작하면 가장 먼저 "chatbot" 노드로 갑니다.
graph_builder.add_edge(START, "chatbot")
# 설계도를 실제 작동 가능한 프로그램(graph)으로 최종 조립(compile)합니다.
graph = graph_builder.compile()


# ============================================================================
# [다양한 그래프 실행 방식 비교]
# 1. invoke / ainvoke : 모든 과정이 끝날 때까지 기다렸다가 완성된 전체 결과를 한 번에 받기 (완제품 택배 받기)
# 2. stream / astream : 각 노드가 끝날 때마다 중간 진행 상황을 실시간으로 받기 (택배 배송 조회하기)
# ============================================================================

def invoke():
    """동기 실행(invoke): 그래프가 끝날 때까지 기다렸다가 전체 대화 기록을 출력합니다."""
    response = graph.invoke(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        }
    )

    # 최종 상태에 담긴 모든 메시지(질문 -> 도구호출 -> 검색결과 -> 최종답변)를 보기 좋게 출력합니다.
    for msg in response["messages"]:
        msg.pretty_print()


async def ainvoke():
    """비동기 실행(ainvoke): 여러 요청을 동시에 처리할 때 유리한 비동기 방식으로 전체 결과를 받습니다."""
    response = await graph.ainvoke(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        }
    )

    for msg in response["messages"]:
        msg.pretty_print()


def stream():
    """기본 스트리밍(stream_mode='updates'): 각 노드가 작업을 마칠 때마다 그 노드가 새로 만든 결과만 출력합니다."""
    response = graph.stream(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        }
    )
    for chunk in response:
        for node, state in chunk.items():
            print("---", node, "---")
            print(state)
            print("=" * 60)

def stream_values():
    """전체 상태 스트리밍(stream_mode='values'): 각 단계가 끝날 때마다 지금까지 누적된 전체 메모장(State)을 출력합니다."""
    response = graph.stream(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        },
        stream_mode="values"
    )

    for chunk in response:
        for state_key, state_value in chunk.items():
            print("--- 현재 상태 ---")
            for msg in state_value:
                print(f"{type(msg).__name__}: {msg.content[:50]}")
            if state_key == "messages":
                state_value[-1].pretty_print()
            print("=" * 60)

def stream_messages():
    """토큰 단위 스트리밍(stream_mode='messages'): 챗GPT처럼 AI가 글자(토큰)를 생성하는 실시간으로 한 조각씩 출력합니다."""
    response = graph.stream(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        },
        stream_mode="messages"
    )

    for token, metadata in response:
        print(token.content)
        # print(metadata["langgraph_node"])

async def astream():
    """비동기 스트리밍(astream): 비동기 방식으로 각 노드의 중간 결과를 실시간으로 받아옵니다."""
    response = graph.astream(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        }
    )
    async for chunk in response:
        for node, state in chunk.items():
            print("---", node, "---")
            print(state)
            print("=" * 60)


if __name__ == "__main__":
    import asyncio
    # 기본으로 비동기 실행(ainvoke)을 테스트합니다. 다른 방식을 체험하려면 아래 주석을 하나씩 해제해 보세요!
    asyncio.run(ainvoke())
    # asyncio.run(astream())

    # invoke()
    # stream()
    # stream_values()
    # stream_messages()
