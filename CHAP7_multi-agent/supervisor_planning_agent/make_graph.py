# ============================================================================
# [7.6 기획(Planning) 기반 슈퍼바이저] 전체 멀티 에이전트 그래프 조립 (make_graph.py)
# 비유: 수석 기획자(planning) ↔ 현장 매니저(supervisor) ↔ 실무자(research, canvas)가
#       체계적인 분업으로 보고서를 완성해내는 '기획형 멀티 에이전트 조직도'를 완성합니다.
# ============================================================================
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

from planning_agent import planning_node
from supervisor_agent import supervisor_node
from canvas_agent import canvas_node
from research_agent import research_node
from settings import State

# [ 1 ] 입력은 간단한 MessagesState로 받고, 내부 및 출력 상태는 계획 정보가 포함된 State를 사용합니다.
graph_builder = StateGraph(State, input_schema=MessagesState, output_schema=State) # [ 1 ]

# [ 2 ] 각 노드를 등록하고 Command(goto=...)로 이동할 수 있는 행선지(destinations)를 지정합니다.
graph_builder.add_node("planning", planning_node, destinations=["supervisor", END]) # [ 2 ]
graph_builder.add_node("supervisor", supervisor_node, destinations=["canvas", "research", "planning"])
graph_builder.add_node("canvas", canvas_node, destinations=["supervisor"])
graph_builder.add_node("research", research_node, destinations=["supervisor"])

# [ 3 ] 시작(START)하면 가장 먼저 수석 기획자(planning)가 전체 계획을 세우도록 연결합니다.
graph_builder.add_edge(START, "planning") # [ 3 ]

# 그래프를 컴파일합니다.
graph = graph_builder.compile()
