# ============================================================================
# [7.5 핸드오프 도구 기반 3중 슈퍼바이저] 웹 검색 전문 에이전트 (web_agent.py)
# 비유: 사내 문서나 FAQ에 없는 외부 최신 뉴스/트렌드를 인터넷(Tavily)에서 빠르게 조사해
#       팀장님(Supervisor)에게 보고하는 '외부 동향 리서치 팀원'입니다.
# ============================================================================
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.graph import END
from langgraph.types import Command
from langchain.messages import HumanMessage
from handoff_tools import create_handoff_messages
from settings import AgentState, get_model

# [ 1 ] Gemini 3.8 Flash 모델과 Tavily 웹 검색 도구를 결합하여 에이전트를 생성합니다.
model = get_model(model_name="gemini-3.8-flash") # [ 1 ]
tavily_search = TavilySearch(max_results=3)

web_agent = create_agent(
    model=model,
    tools=[tavily_search],
)

def create_web_agent(state: AgentState) -> Command: # [ 2 ]
    """슈퍼바이저가 전달한 query를 받아 웹 검색을 수행하고 결과를 반환하는 노드 함수입니다."""
    # 슈퍼바이저의 핸드오프 도구가 state["query"]에 넣어준 세부 질문을 꺼냅니다.
    query = state.get("query", "")
    agent_state = {"messages": [HumanMessage(content=query)]}

    result = web_agent.invoke(agent_state)

    # [ 3 ] 작업 완료 후 슈퍼바이저 복귀를 알리는 표식 메시지(AIMessage, ToolMessage)를 생성합니다.
    ai_message, tool_message = create_handoff_messages("web_search") # [ 3 ]

    result["messages"].extend([ai_message, tool_message])

    # [ 4 ] 결과를 상태에 업데이트하고 현재 서브 노드를 종료합니다(메인 그래프의 엣지를 통해 supervisor로 복귀).
    return Command( # [ 4 ]
        update={"messages": result["messages"]},
        goto=END,
    )
