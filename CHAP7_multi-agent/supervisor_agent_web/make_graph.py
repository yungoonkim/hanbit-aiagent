# ============================================================================
# [7.4 슈퍼바이저 패턴 기초] 전체 그래프 연결 및 컴파일 (make_graph.py)
# 비유: 중앙의 팀장님(supervisor)을 중심으로 웹 분석 팀원(web_agent)과
#       DB 사서 팀원(database_agent)이 별 모양(Star Topology)으로 연결된 조직도를 완성합니다.
# ============================================================================
from langgraph.graph import StateGraph, MessagesState, START
from supervisor_agent import supervisor_node
from web_agent import web_node
from database_agent import database_node

# 전체 대화를 관리하는 StateGraph를 생성합니다.
graph_builder = StateGraph(MessagesState)

# 시작(START)하면 무조건 팀장님(supervisor)이 먼저 요청을 받습니다.
graph_builder.add_edge(START, "supervisor")

# 팀장님과 두 명의 전문 팀원 노드를 등록합니다.
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("web_agent", web_node)

# web_agent(서브 그래프)는 작업이 끝나면 다시 팀장님(supervisor)에게 보고하도록 엣지를 연결합니다.
# (database_node는 내부에서 Command(goto="supervisor")를 반환하므로 자동 복귀합니다.)
graph_builder.add_edge("web_agent", "supervisor")
graph_builder.add_node("database_agent", database_node)

# 그래프를 컴파일합니다.
graph = graph_builder.compile()
