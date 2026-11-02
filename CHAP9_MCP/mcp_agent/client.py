# ============================================================================
# [9장 실습 1] MCP 클라이언트 만들기 (서버의 도구와 프롬프트를 연결해 쓰는 AI 에이전트)
# 💡 일상 비유: server.py가 'USB-C 외장 도구 상자'라면, client.py는 그 도구 상자를 선(stdio)으로
# 꽂아서 실제로 사용하는 '노트북 본체(AI 에이전트)'입니다.
# ============================================================================
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv

# .env 파일에서 GOOGLE_API_KEY 등 환경 변수를 불러옵니다.
load_dotenv()

# 두뇌 역할을 할 Gemini 모델(gemini-3.8-flash)을 준비합니다.
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# uv run 으로 어디서 실행하든 현재 client.py가 위치한 폴더의 server.py와
# 현재 가상환경의 파이썬(sys.executable)을 정확히 바라보도록 설정합니다.
SERVER_SCRIPT = str(Path(__file__).resolve().parent / "server.py")

# StdioServerParameters: "어떤 프로그램(파이썬)으로 어떤 서버 파일(server.py)을 실행해서 연결할지" 적는 연결 설정표입니다.
server_params = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
)


async def run():
    # stdio_client로 server.py를 백그라운드에서 실행하고, 데이터를 주고받을 읽기(read)/쓰기(write) 통로를 엽니다.
    async with stdio_client(server_params) as (read, write):
        # ClientSession으로 서버와 대화 세션을 만들고 초기화(악수/연결 확인)합니다.
        async with ClientSession(read, write) as session:
            await session.initialize()

            ##### AGENT #####
            # [ 1 ] MCP 서버에 등록된 모든 도구(read_user_info, save_diary)를 랭체인 도구 형태로 불러옵니다.
            tools = await load_mcp_tools(session) # [ 1 ]
            # [ 2 ] 모델, MCP 도구, 그리고 대화 맥락을 기억할 단기 메모리(InMemorySaver)를 합쳐 에이전트를 생성합니다.
            agent = create_agent(model, tools, checkpointer=InMemorySaver()) # [ 2 ]

            ##### REQUEST & REPOND #####
            # [ 3 ] 첫 번째 대화일 때만 MCP 서버에 저장된 기본 프롬프트(default_prompt)를 불러오기 위한 스위치입니다.
            is_first_input = True # [ 3 ]
            while True:
                user_input = input("질문을 입력하세요: ")

                # q, exit, quit를 입력하면 대화를 종료합니다.
                if user_input.lower() in ["q", "exit", "quit"]:
                    break

                if is_first_input:
                    # 첫 입력 시: 서버의 @mcp.prompt()로 정의된 '친절한 개인 비서 매뉴얼 + 첫 인사'를 가져옵니다.
                    prompts = await load_mcp_prompt(
                        session, "default_prompt", arguments={"message": user_input}
                    )
                    is_first_input = False
                else:
                    # 두 번째 입력부터는 이미 단기 메모리(InMemorySaver)에 역할이 기억되어 있으므로 사용자 메시지만 전달합니다.
                    prompts = [{"role": "user", "content": user_input}]

                # [ 4 ] 비동기 방식(ainvoke)으로 에이전트에게 질문을 던지고 응답을 받습니다.
                # thread_id="1" 덕분에 대화가 이어지는 동안 앞선 대화 내용을 모두 기억합니다!
                response = await agent.ainvoke( # [ 4 ]
                    {
                        "messages": prompts
                    },
                    config={"configurable": {"user_id": "user_123", "thread_id": "1"}}
                )

                # 주고받은 메시지 내역을 보기 좋게 출력합니다.
                for msg in response["messages"]:
                    msg.pretty_print()
                # print(response["messages"][-1].content)


import asyncio
# 비동기 함수 run()을 실행합니다.
asyncio.run(run())
