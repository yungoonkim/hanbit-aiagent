# ============================================================================
# [9장 실습 2] 여러 개의 MCP 서버를 동시에 연결하는 멀티 에이전트 (MultiServerMCPClient)
# 💡 일상 비유: 노트북에 'USB-C 멀티허브(MultiServerMCPClient)'를 꽂아서
# 1번 포트에는 인터넷 검색 장비(Tavily 원격 MCP 서버, HTTP 방식)를 연결하고,
# 2번 포트에는 내 컴퓨터 파일 정리 장비(로컬 server.py, stdio 방식)를 동시에 연결해
# 팀장님(Supervisor)이 상황에 맞는 담당 직원에게 일을 배분하는 시스템입니다!
# ============================================================================
from dotenv import load_dotenv
import os
import asyncio
import sys

from typing import Literal
from typing_extensions import TypedDict

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


import sys, asyncio
# 윈도우(Windows) 환경에서 비동기(asyncio) 실행 시 발생할 수 있는 이벤트 루프 오류를 방지합니다.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# .env 파일에서 환경 변수(GOOGLE_API_KEY, TAVILY_API_KEY)를 불러옵니다.
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

# 슈퍼바이저와 작업자 에이전트들이 공통으로 사용할 Gemini 모델을 초기화합니다.
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

######################## GRAPH STATE ########################

# 슈퍼바이저(팀장)가 다음 순번으로 누구를 부를지 결정할 때 사용하는 라우팅 도구 양식입니다.
class Router(TypedDict):
    """라우팅 할 작업자
    file_searcher: 파일 정보 호출 및 생성 작업자
    web_searcher: 웹 검색 작업자
    """

    next: Literal["file_searcher", "web_searcher"]


# 그래프 전체에서 공유할 상태(대화 메시지 목록 + 다음 작업자 이름)를 정의합니다.
class State(MessagesState):
    next: str


async def run():
    # 상대 경로("./server.py")가 항상 이 파일이 있는 폴더를 기준으로 작동하도록 작업 위치를 맞춥니다.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # MultiServerMCPClient: 여러 대의 MCP 서버(원격 HTTP 서버 + 로컬 stdio 서버)를 한 번에 묶어 관리하는 멀티허브입니다!
    client = MultiServerMCPClient(
        {
            # [ 1 ] 첫 번째 서버: 인터넷을 통해 접속하는 Tavily 웹 검색 MCP 서버 (streamable_http 통신)
            "tavily": { # [ 1 ]
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
                "transport": "streamable_http",
            },
            # [ 2 ] 두 번째 서버: 내 컴퓨터의 파이썬으로 직접 실행하는 로컬 파일 관리 MCP 서버 (stdio 통신)
            "file": { # [ 2 ]
                "transport": "stdio",
                "command": sys.executable,
                "args": ["./server.py"],
            }
        }
    )

    ######################## CHECK TOOLS ########################
    # 연결된 모든 MCP 서버의 도구를 한꺼번에 가져와 확인해 봅니다.
    all_tools = await client.get_tools()
    print(f"Total tools loaded: {len(all_tools)}")
    print(f"Tool names: {[tool.name for tool in all_tools]}")

    # 각 작업자에게 필요한 도구만 나눠주기 위해 서버 이름(server_name)별로 도구를 따로 가져옵니다.
    tavily_tools = await client.get_tools(server_name="tavily")
    file_tools = await client.get_tools(server_name="file")

    print(f"Tavily tools: {[tool.name for tool in tavily_tools]}")
    print(f"File tools: {[tool.name for tool in file_tools]}")

    ######################## AGENT ########################
    # [ 1 ] 팀에 소속된 두 명의 전문 작업자(파일 담당, 웹 검색 담당) 명단입니다.
    members = ["file_searcher", "web_searcher"] # [ 1 ]

    # 슈퍼바이저(팀장)에게 부여할 업무 지시서(시스템 프롬프트)입니다.
    system_prompt = f"""
당신은 다음 작업자들 간의 대화를 관리하는 슈퍼바이저입니다.
작업자들은 다음과 같습니다: {members}
특정 작업자가 수행할 작업이 있다면, Router 도구를 사용해 다음 작업자를 지정하세요.
작업이 필요없거나 즉시 답변하고, 이미 작업을 완료했다면 최종 답변을 반환하세요.
"""

    ######################## SUPERVISOR ########################
    # [ 2 ] 슈퍼바이저 노드: 사용자의 요청과 진행 상황을 보고 다음 작업자를 호출하거나 대화를 마무리(END)합니다.
    async def supervisor_node( # [ 2 ]
        state: State,
    ) -> Command[Literal["file_searcher", "web_searcher", END]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        # [ 3 ] 모델에 Router 도구를 쥐여주어, 작업자 호출이 필요할 때 도구 호출(tool_calls)을 생성하게 합니다.
        response = model.bind_tools([Router]).invoke(messages) # [ 3 ]

        # [ 4 ] 만약 모델이 Router 도구를 호출했다면, 지정된 작업자(file_searcher 또는 web_searcher)에게 이동(goto)합니다.
        if hasattr(response, "tool_calls") and len(response.tool_calls) > 0: # [ 4 ]
            goto = response.tool_calls[0]["args"]["next"]
            return Command(goto=goto, update={"next": goto})

        # [ 5 ] 더 이상 부를 작업자가 없다면, 슈퍼바이저의 최종 답변을 메시지에 담고 그래프를 종료(END)합니다.
        else: # [ 5 ]
            final_message = AIMessage(content=response.content, name="supervisor")
            return Command(
                goto=END,
                update={"messages": [final_message]}
            )

    ######################## 1) FILE SEARCHER ########################
    # [ 1 ] 로컬 파일 MCP 도구(file_tools)만 전문으로 다루는 파일 검색 에이전트를 만듭니다.
    file_searcher = create_agent(model, file_tools) # [ 1 ]

    # [ 2 ] 파일 검색 작업자 노드: 임무를 수행한 뒤 결과 보고서를 들고 다시 슈퍼바이저(팀장)에게 돌아갑니다.
    async def file_search_node(state: State) -> Command[Literal["supervisor"]]: # [ 2 ]
        result = await file_searcher.ainvoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content, name="file_searcher"
                    )
                ]
            },
            goto="supervisor",
        )

    ######################## 2) WEB SEARCHER (TAVILY) ########################
    # [ 3 ] Tavily 웹 검색 MCP 도구(tavily_tools)만 전문으로 다루는 웹 검색 에이전트를 만듭니다.
    web_searcher = create_agent(model, tavily_tools) # [ 3 ]

    # [ 4 ] 웹 검색 작업자 노드: 인터넷 검색을 마친 뒤 결과 보고서를 들고 다시 슈퍼바이저(팀장)에게 돌아갑니다.
    async def web_search_node(state: State) -> Command[Literal["supervisor"]]: # [ 4 ]
        result = await web_searcher.ainvoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content, name="web_searcher"
                    )
                ]
            },
            goto="supervisor",
        )

    ######################## GRAPH COMPILE ########################
    # 설계도(StateGraph)에 슈퍼바이저와 두 작업자 노드를 등록하고, 단기 메모리(InMemorySaver)와 함께 컴파일합니다.
    graph_builder = StateGraph(State)
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("file_searcher", file_search_node)
    graph_builder.add_node("web_searcher", web_search_node)
    memory = InMemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    ######################## REQUEST & REPOND ########################
    # [ 1 ] 대화 맥락을 유지하기 위해 thread_id를 "1"로 설정합니다.
    config = {"configurable": {"thread_id": "1"}} # [ 1 ]
    while True:
        try:
            user_input = input("질문을 입력하세요: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("안녕히 가세요!")
                break

            # [ 2 ] astream을 사용해 각 노드(슈퍼바이저 -> 작업자 -> 슈퍼바이저)가 처리되는 과정을 실시간으로 스트리밍합니다.
            async for _, chunk in graph.astream( # [ 2 ]
                {"messages": user_input},
                stream_mode="updates",
                subgraphs=True,
                config=config,
            ):
                # [ 3 ] 단계별 업데이트 결과에서 가장 최근 메시지를 꺼내 화면에 출력합니다.
                for _, node_chunk in chunk.items(): # [ 3 ]
                    if "messages" in node_chunk:
                        node_chunk["messages"][-1].pretty_print()
                    else:
                        print(node_chunk)

        except Exception as e:
            print(f"종료합니다.{e}")
            break


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
