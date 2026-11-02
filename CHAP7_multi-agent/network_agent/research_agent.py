# ============================================================================
# [7.3 네트워크 멀티 에이전트] 자료 조사 전문 에이전트 (research_agent.py)
# 비유: 신문사에서 취재만 전담하는 '취재 기자'입니다.
#       인터넷에서 자료를 모은 뒤, 그래프나 시각화가 필요하면 '디자이너(html_generator)'에게
#       바통을 넘기고, 조사가 끝났으면 '최종 답변'을 붙여 일을 마무리합니다.
# ============================================================================
from typing import Literal

from langchain_tavily import TavilySearch
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.types import Command
from langgraph.graph import MessagesState, END

from settings import get_model, make_system_prompt

from dotenv import load_dotenv

# .env 파일에서 API 키(GOOGLE_API_KEY, TAVILY_API_KEY 등)를 불러옵니다.
load_dotenv()

# Gemini 3.8 Flash 모델과 Tavily 웹 검색 도구를 준비합니다.
llm = get_model("gemini-3.8-flash")
tool = TavilySearch(max_results=5)

# 웹 검색 전용 에이전트를 생성합니다.
research_agent = create_agent(
    llm,
    tools=[tool],
    system_prompt=make_system_prompt(
        "당신은 웹검색을 통한 자료조사만 할 수 있습니다. 차트 시각화(html 파일 생성)를 담당하는 에이전트와 함께 일하고 있으니 차트 생성이나 파일 생성은 수행하지 말고 넘기세요."
    ),
)

def research_node(state: MessagesState) -> Command[Literal["html_generator", END]]:
    """
    자료 조사를 수행하고, 상황에 따라 HTML 생성 에이전트로 넘기거나 종료(END)하는 노드입니다.
    """
    # [ 1 ] 현재 대화 상태(state)를 바탕으로 리서치 에이전트를 실행합니다.
    result = research_agent.invoke(state) # [ 1 ]

    # [ 2 ] 다음 에이전트가 "아, 이건 researcher 동료가 보내준 조사 결과구나!"라고 알 수 있도록
    #       마지막 메시지를 name="researcher"가 적힌 HumanMessage로 포장합니다.
    result["messages"][-1] = HumanMessage( # [ 2 ]
        content=result["messages"][-1].content, name="researcher"
    )

    # [ 3 ] 메시지에 '최종 답변'이라는 암호가 포함되어 있으면 모든 업무가 끝난 것이므로 END로 가고,
    #       아직 차트/HTML 제작이 남았다면 동료인 'html_generator'에게 바통을 넘깁니다.
    last_message = result["messages"][-1] # [ 3 ]
    if "최종 답변" in last_message.content:
        goto = END
    else:
        goto = "html_generator"

    # [ 4 ] 업데이트된 대화 기록과 다음 이동할 목적지(goto)를 Command에 담아 반환합니다.
    return Command( # [ 4 ]
        update={"messages": result["messages"]},
        goto=goto,
    )
