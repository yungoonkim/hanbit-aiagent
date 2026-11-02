# ============================================================================
# [6.3절] 코딩 에이전트 만들기 (coding_agent/agent.py)
# ============================================================================
# 이 파일은 랭체인의 `create_agent` 함수를 사용해 단 몇 줄만으로
# "코드 작성 -> 실행 테스트 -> .py 파일 저장"을 스스로 수행하는 코딩 에이전트를 만듭니다.
#
# 💡 비전공자를 위한 일상 비유:
# - 6.2절에서는 우리가 직접 StateGraph, BasicToolNode, route_tools를 하나하나 조립(수동 변속기 자동차)했다면,
# - 6.3절의 `create_agent`는 "두뇌(model)와 연장통(tools)만 넣어주면 알아서 에이전트 그래프를 완성해 주는"
#   자동 변속기(오토매틱) 자동차입니다!
# ============================================================================

from dotenv import load_dotenv

# 단 한 줄로 도구 호출형 에이전트 그래프를 만들어주는 create_agent를 가져옵니다.
from langchain.agents import create_agent, AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic

# 방금 tools.py에서 만든 '파이썬 실행 도구'와 '파일 저장 도구'를 가져옵니다.
from tools import python_exec_tool, file_write_tool

# .env 파일에서 API 키를 불러옵니다.
load_dotenv()

# AI에게 쥐여줄 두 가지 연장(코드 실행기, 파일 저장기)을 리스트에 담습니다.
# tools = [python_exec_tool]
tools = [python_exec_tool, file_write_tool]

# 코딩 에이전트의 두뇌가 될 Gemini 3.8 Flash 모델을 준비합니다.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
# llm = ChatAnthropic(model="claude-opus-4-1-20250805") # 책의 실행 결과는 본 클로드 모델을 사용하여 나온 결과입니다.

# create_agent: 모델과 도구만 전달하면 내부적으로 [모델 -> 도구 실행 -> 모델] 순환 그래프를 자동 생성합니다!
graph = create_agent(
    model=llm,
    tools=tools
)


if __name__ == "__main__":
    # 에이전트에게 2가지 임무가 담긴 지시사항을 내리고 실시간(stream)으로 작업 과정을 지켜봅니다.
    # 1) 피보나치 수열 코드를 짜서 python_exec_tool로 실행 확인하기
    # 2) 잘 작동하면 file_write_tool로 .py 파일로 저장하기
    response = graph.stream(
        {
            "messages": [
                "첫번째 항이 1인 피보나치 수열을 출력하는 파이썬 코드를 작성해주세요. 정상적으로 실행되는 지 확인도 해주세요. ",
                "확인했다면 그 코드는 .py 파일로 저장하세요."
            ]
        }
    )

    # 각 노드(model, tools)가 일할 때마다 그 결과를 화면에 순서대로 출력합니다.
    for chunk in response:
        for node, value in chunk.items():
            if node:
                print("---", node, "---")
            if "messages" in value:
                print(value['messages'][0].content)
