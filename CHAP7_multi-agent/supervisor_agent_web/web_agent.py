# ============================================================================
# [7.4 슈퍼바이저 패턴 기초] 웹사이트 크롤링 및 확장 질문 생성 에이전트 (web_agent.py)
# 비유: 사용자가 건네준 인터넷 주소(URL)에 직접 방문해 불필요한 광고를 걷어내고 본문만 깔끔히 요약한 뒤,
#       "이런 질문도 더 궁금하지 않으세요?"라며 확장 질문까지 뽑아오는 '웹 분석 담당 팀원'입니다.
# ============================================================================
from typing import List
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from settings import get_model

# 크롤링한 웹 페이지 데이터를 노드 간에 임시 공유하기 위한 전역 리스트입니다.
_loaded_web_content = []

def _clean_content(content: str) -> str:
    """
    웹 콘텐츠에서 불필요한 요소(광고, 메뉴, 푸터 등)를 제거하고 핵심 본문만 깔끔하게 정리합니다.

    Args:
        content: 정리할 원본 콘텐츠

    Returns:
        정리된 콘텐츠
    """

    llm = get_model("gemini-3.8-flash")

    cleaning_prompt = f"""다음 웹페이지에서 추출한 텍스트를 정리해주세요.

    원본 텍스트:
    {content}

    지시사항:
    1. 불필요한 공백, 개행 문자를 정리하여 읽기 쉽게 만들기
    2. 광고 텍스트, 네비게이션 메뉴, 푸터 정보 등 본문과 무관한 내용 삭제
    3. 핵심 내용만 남기고 중복되거나 불필요한 텍스트 제거
    4. 문단 구조를 유지하면서 정돈된 형태로 반환
    5. 중요한 정보(제목, 본문 내용, 핵심 데이터)는 반드시 유지

    정리된 텍스트만 반환해주세요 (추가 설명 없이):"""

    response = llm.invoke(cleaning_prompt)
    cleaned_content = response.content.strip()
    return cleaned_content

@tool
def web_content_loader(
    urls: List[str]
):
    """
    WebBaseLoader를 사용하여 웹사이트의 내용을 가져오는 도구입니다.

    Args:
        urls (List[str]): 크롤링할 웹사이트 URL들 목록

    Returns:
        str: 각 웹사이트의 제목과 정리된 내용
    """
    try:
        # [ 1 ] WebBaseLoader로 전달받은 URL 목록의 웹페이지 본문을 읽어옵니다.
        loader = WebBaseLoader(urls) # [ 1 ]
        all_docs = loader.load()

        if not all_docs:
            return "웹사이트 내용을 가져올 수 없습니다. URL을 확인해주세요."

        result = []
        # [ 2 ] 각 웹페이지의 원본 텍스트를 LLM으로 정제(_clean_content)하여 리스트에 담습니다.
        for doc in all_docs: # [ 2 ]
            cleaned_content = _clean_content(doc.page_content)

            result.append({
                "url": doc.metadata.get('source', 'No url'),
                "title": doc.metadata.get('title', 'No title'),
                "full_content": cleaned_content,
                "content_length": len(cleaned_content)
            })

        # [ 3 ] 다음 단계(확장 질문 생성 노드)에서 활용할 수 있도록 전역 변수에 저장합니다.
        global _loaded_web_content # [ 3 ]
        _loaded_web_content = result

        return result

    except Exception as e:
        return f"웹사이트 크롤링에 실패했습니다. 오류: {repr(e)}"


def chatbot(state: MessagesState):
    """사용자 메시지에서 URL을 추출하여 web_content_loader 도구 호출을 결정하는 노드입니다."""
    # [ 4 ] 시스템 프롬프트와 현재까지의 대화 내역을 결합합니다.
    messages = [ # [ 4 ]
        {
            "role": "system",
            "content": """
            당신은 URL 추출과 웹 크롤링 전문가입니다.
            사용자의 메시지에서 URL을 찾아내고, web_content_loader 도구를 사용하여 해당 웹페이지의 내용을 가져오세요.
            여러 개의 URL을 발견한 경우, 반드시 모든 URL을 하나의 리스트로 묶어서 web_content_loader를 한번만 호출하세요.
            """
        },
    ] + state["messages"]

    llm = get_model("gemini-3.8-flash")
    # [ 5 ] 도구를 바인딩한 모델을 호출하여 URL 크롤링 도구 호출(tool_calls)을 생성합니다.
    result = llm.bind_tools([web_content_loader]).invoke(messages) # [ 5 ]

    return {"messages": [result]}


def question_generator_node(state: MessagesState):
    """로드된 웹 페이지의 내용을 기반으로 사용자가 궁금해할 만한 확장 질문을 생성하는 노드입니다."""
    global _loaded_web_content

    # [ 1 ] 크롤링하여 정제해 둔 웹 콘텐츠들을 하나의 문자열로 합칩니다.
    combined_content = "" # [ 1 ]
    for content_item in _loaded_web_content:
        if isinstance(content_item, dict):
            url = content_item.get("url", "Unknown URL")
            title = content_item.get("title", "No title")
            full_content = content_item.get("full_content", "")  # 정제된 콘텐츠 사용
            combined_content += f"URL: {url}\n제목: {title}\n\n{full_content}\n\n{'='*80}\n\n"
    # [ 2 ] 본문 내용을 바탕으로 URL별 1~2개의 후속 확장 질문을 생성하도록 요청합니다.
    prompt = f"""
    다음 웹 페이지 내용을 바탕으로, 사용자가 관심 있을만한 확장 질문을 생성해주세요.

    웹 페이지 내용:
    {combined_content}

    요구사항:
    1. 웹 페이지 내용을 기반으로 추가로 궁금한 질문들
    2. 한국어로 작성
    3. 각 질문은 번호를 매겨서 url과 함께 딕셔너리 형태로 제공
    4. url 별로 1-2 개의 질문을 생성
    """
    llm = get_model("gemini-3.8-flash")
    response = llm.invoke(prompt)
    questions = response.content

    result_message = f"웹 페이지 내용을 기반으로 사용자에게 다음 질문을 추가로 생각해보도록 제안할 수 있습니다. : \n\n{questions}\n"

    # [ 3 ] 발신자 이름(name="web_agent")을 명시하여 슈퍼바이저가 누가 보고했는지 알 수 있게 합니다.
    return {"messages": [AIMessage(content=result_message, name="web_agent")]} # [ 3 ]


def create_workflow() -> CompiledStateGraph:
    """
    웹 에이전트 내부의 서브 그래프(소규모 작업 흐름도)를 조립하여 반환합니다.
    흐름: START -> chatbot -> (도구 호출 시) url_extractor -> question_generator -> END
    """
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=[web_content_loader])
    graph_builder.add_node("url_extractor", tool_node)
    graph_builder.add_node("question_generator", question_generator_node)

    # [ 1 ] 시작하자마자 URL을 찾는 chatbot 노드로 이동합니다.
    graph_builder.add_edge(START, "chatbot") # [ 1 ]
    # [ 2 ] 도구 호출이 있으면 url_extractor로, 없으면 바로 종료(END)합니다.
    graph_builder.add_conditional_edges( # [ 2 ]
        "chatbot",
        tools_condition,
        {"tools": "url_extractor", END: END}
    )
    # [ 3 ] 웹 본문 추출이 끝나면 확장 질문 생성 노드로 이어집니다.
    graph_builder.add_edge("url_extractor", "question_generator") # [ 3 ]
    # [ 4 ] 질문 생성까지 마치면 서브 그래프 작업이 완료됩니다.
    graph_builder.add_edge("question_generator", END) # [ 4 ]

    return graph_builder.compile()

# 메인 그래프에 노드로 등록할 컴파일된 웹 에이전트 워크플로우입니다.
web_node = create_workflow()
