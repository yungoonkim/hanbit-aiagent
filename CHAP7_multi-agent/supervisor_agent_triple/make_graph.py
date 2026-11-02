# ============================================================================
# [7.5 핸드오프 도구 기반 3중 슈퍼바이저] 전체 멀티 에이전트 그래프 조립 (make_graph.py)
# 비유: 본부장(supervisor)을 중심에 두고 3개의 실무팀(web_search, db_search, faq)이
#       업무를 받아 처리한 뒤 다시 본부장에게 결과를 보고하는 순환 조직도를 만듭니다.
# ============================================================================
from langgraph.graph import StateGraph, MessagesState, END
from supervisor_agent import supervisor, agents

# [ 1 ] 메시지 상태를 기반으로 하는 StateGraph 빌더를 생성합니다.
graph_builder = StateGraph(MessagesState) # [ 1 ]

# [ 2 ] 슈퍼바이저 노드를 추가하고, 이동 가능한 목적지(destinations)를 명시합니다.
agent_names = list(agents.keys()) # [ 2 ]
graph_builder.add_node(
    "supervisor",
    supervisor,
    destinations=tuple(agent_names) + (END,)
)

# [ 3 ] 3개의 전문 하위 에이전트 노드를 그래프에 등록합니다.
graph_builder.add_node("web_search", agents["web_search"]) # [ 3 ]
graph_builder.add_node("db_search", agents["db_search"])
graph_builder.add_node("faq", agents["faq"])

# 그래프의 시작점을 슈퍼바이저로 지정합니다.
graph_builder.set_entry_point("supervisor")

# [ 4 ] 모든 하위 에이전트가 작업을 마치면 자동으로 슈퍼바이저에게 복귀하도록 엣지를 연결합니다.
for agent_name in agents.keys(): # [ 4 ]
    graph_builder.add_edge(agent_name, "supervisor")

# 최종 실행 가능한 그래프를 컴파일합니다.
graph = graph_builder.compile()
