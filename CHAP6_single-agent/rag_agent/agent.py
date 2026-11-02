# ============================================================================
# [6.5절] Self-RAG 에이전트 그래프 조립 및 실행 (rag_agent/agent.py)
# ============================================================================
# 이 파일은 앞서 만든 상태(state.py), 5개 작업 노드(nodes.py), 2개 심사관 엣지(edges.py)를
# 하나의 완벽한 순서도(StateGraph)로 연결하고 실행하는 최종 본체 파일입니다.
#
# 💡 전체 흐름 한눈에 보기:
# [START] -> [chatbot] --(검색 필요 시)--> [retriever] -> [context_organizer]
#    |--(1차 심사: 문서가 질문과 관련 있나?)
#         |-- No (관련 없음) --> [transform_query (질문 교정)] --> [retriever (재검색!)]
#         |-- Yes (관련 있음) -> [generate (답변 작성)]
#                                  |--(2차 심사: 문서에 근거한 답변인가?)
#                                       |-- No (환각/거짓말) --> [generate (재작성!)]
#                                       |-- Yes (근거 확실)  --> [END (완료!)]
# ============================================================================

from dotenv import load_dotenv

load_dotenv()

# 랭그래프가 기본 제공하는 도구 호출 판별기(tools_condition)를 가져옵니다.
# (6.2절에서 우리가 직접 짰던 route_tools 함수와 똑같은 역할을 해주는 편리한 내장 함수입니다!)
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END

from state import AgentState
from nodes import chatbot, retrieve, context_organizer, generate, transform_query
from edges import decide_to_generate, check_hallucinations


# 1. 그래프 설계도 생성: 내부에서는 AgentState를 쓰고, 사용자 입력은 간단한 MessagesState로 받습니다.
graph_builder = StateGraph(AgentState, input_schema=MessagesState)

# 2. 안내 데스크(chatbot)와 도서관 사서(retriever) 노드 등록
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("retriever", retrieve)

# 3. 출발점(START) -> 안내 데스크(chatbot) 연결 후, 검색 도구 호출 여부에 따른 갈림길 연결
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "retriever",  # 검색 도구를 호출했다면 사서(retriever)에게 이동
        END: END,              # 일상 대화라서 바로 답했다면 즉시 종료(END)
    }
)

# 4. 자료 정리원, 질문 교정가, 최종 집필가 노드 등록
graph_builder.add_node("context_organizer", context_organizer)
graph_builder.add_node("transform_query", transform_query)
graph_builder.add_node("generate", generate)

# 5. 검색 후 자료 정리원으로 이동 -> 1차 심사관(decide_to_generate)이 관련성 평가!
graph_builder.add_edge("retriever", "context_organizer")
graph_builder.add_conditional_edges(
    "context_organizer",
    decide_to_generate,
    {
        "transform_query": "transform_query",  # 관련 없으면 질문 교정 노드로!
        "generate": "generate",                # 관련 있으면 답변 생성 노드로!
    },
)

# 6. 질문을 교정했으면 다시 사서(retriever)에게 가서 재검색!
graph_builder.add_edge("transform_query", "retriever")

# 7. 답변을 생성했으면 2차 심사관(check_hallucinations)이 환각 여부 평가!
graph_builder.add_conditional_edges(
    "generate",
    check_hallucinations,
    {
        "not supported": "generate",  # 문서에 없는 내용이면 다시 답변 작성!
        "support": END                # 문서 사실에 근거한 정확한 답변이면 최종 종료(END)!
    },
)

# 8. 모든 노드와 화살표 조립 완료 (컴파일)
graph = graph_builder.compile()


if __name__ == "__main__":
    # 완성된 그래프 구조를 이미지(graph.png)로 저장해 봅니다.
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception:
        pass

    # 한글 맞춤법/표준어 규정 질문("구개음화가 뭐야?")을 던져 각 부서가 일하는 과정을 스트리밍으로 확인합니다.
    response = graph.stream(
        {
            "messages": [
                "구개음화가 뭐야?"
            ]
        }
    )

    for chunk in response:
        for node, value in chunk.items():
            if node:
                print("---", node, "---")
            if "messages" in value:
                print(value['messages'][0].content)

        print("="*60)
