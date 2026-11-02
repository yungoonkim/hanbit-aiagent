# ============================================================================
# [7.3 네트워크 멀티 에이전트] 그래프 조립 및 컴파일 (make_graph.py)
# 비유: 취재 기자(researcher)와 디자이너(html_generator) 두 명의 전문가를 한 사무실(StateGraph)에
#       배치합니다. 각 노드 내부의 Command가 다음 행선지를 결정하므로 그래프 연결이 매우 심플합니다!
# ============================================================================
from langgraph.graph import StateGraph, MessagesState, START
from network_agent.research_agent import research_node
from network_agent.chart_agent import html_node

# 대화 메시지 기록(MessagesState)을 공유하는 그래프 빌더를 생성합니다.
graph_builder = StateGraph(MessagesState)

# 두 명의 전문가 노드를 그래프에 등록합니다.
graph_builder.add_node("researcher", research_node)
graph_builder.add_node("html_generator", html_node)

# 시작(START)하면 먼저 자료 조사 전문가(researcher)가 업무를 시작하도록 연결합니다.
graph_builder.add_edge(START, "researcher")

# 실행 가능한 멀티 에이전트 그래프로 컴파일합니다.
graph = graph_builder.compile()
