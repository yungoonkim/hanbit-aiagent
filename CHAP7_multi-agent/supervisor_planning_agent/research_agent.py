# ============================================================================
# [7.6 기획(Planning) 기반 슈퍼바이저] 웹 자료 조사 전문 에이전트 (research_agent.py)
# 비유: 기획자가 세운 계획 중 '자료 조사' 단계가 오면 팀장님의 지시를 받아
#       인터넷(Tavily)에서 신뢰할 수 있는 자료와 출처 URL을 수집해 오는 '리서치 전문가'입니다.
# ============================================================================
from typing import Literal
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_tavily import TavilySearch

from settings import get_model, State

# [ 1 ] Tavily 웹 검색 도구와 Gemini 3.8 Flash 모델로 리서치 에이전트를 생성합니다.
tavily_search = TavilySearch(max_results=3) # [ 1 ]
agent = create_agent(
    get_model("gemini-3.8-flash"),
    [tavily_search],
    system_prompt="""당신은 웹 검색과 자료 조사를 전담하는 연구 전문가입니다.

    주요 역할:
    1. 사용자가 요청한 주제에 대한 웹 검색 수행
    2. 관련 웹사이트에서 상세 정보 수집

    항상 정확하고 신뢰할 수 있는 정보 수집에 중점을 두세요."""
)

def research_node(state: State) -> Command[Literal["supervisor"]]: # [ 2 ]
    """Research 에이전트를 실행하고 조사 결과를 담아 팀장님(supervisor)에게 복귀하는 노드입니다."""

    result = agent.invoke(state)
    # 나중에 canvas_agent와 supervisor가 이 메시지를 식별할 수 있도록 이름을 'research_agent'로 지정합니다.
    result["messages"][-1].name = "research_agent"

    return Command(
        goto="supervisor",
        update={"messages": result["messages"]}
    )
