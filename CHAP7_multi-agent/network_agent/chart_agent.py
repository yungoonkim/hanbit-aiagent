# ============================================================================
# [7.3 네트워크 멀티 에이전트] 차트 및 HTML 시각화 전문 에이전트 (chart_agent.py)
# 비유: 취재 기자(researcher)가 모아온 데이터를 받아서 보기 좋은 인포그래픽 웹페이지(.html)로
#       만들어내는 '시각화 디자이너'입니다. 추가 자료가 더 필요하면 다시 기자에게 요청하고,
#       파일 저장이 끝나면 '최종 답변'으로 전체 작업을 종료합니다.
# ============================================================================
from typing import Literal

from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from settings import get_model, make_system_prompt

# Gemini 3.8 Flash 모델을 가져옵니다.
llm = get_model("gemini-3.8-flash")

@tool
def html_generator_tool(
    html_content: str,
    filename: str,
):
    # [ 1 ] LLM이 작성한 HTML/CSS/JS 코드를 실제 파일로 저장해주는 도구입니다.
    """
    HTML 파일을 생성하고 저장합니다. 차트나 데이터 시각화를 HTML/CSS/JavaScript로 생성할 수 있습니다.

    Args:
        html_content (str): HTML 코드 내용
        filename (str): 저장할 HTML 파일명 (예: report.html)

    Returns:
        str: 파일 생성 결과 메시지
    """
    try:
        # [ 2 ] 지정된 파일명으로 UTF-8 인코딩을 사용해 HTML 파일을 저장합니다.
        with open(filename, 'w', encoding='utf-8') as f: # [ 2 ]
            f.write(html_content)
        return f"HTML 파일이 성공적으로 생성되었습니다: {filename}\n파일을 브라우저에서 열어 확인할 수 있습니다.\n\n작업이 완료되었다면 '최종 답변'으로 응답하세요."
    except Exception as e:
        return f"HTML 파일 생성에 실패하였습니다. 오류: {repr(e)}"

# [ 3 ] HTML 생성 도구를 장착한 시각화 전문 에이전트를 생성합니다.
html_agent = create_agent( # [ 3 ]
    llm,
    [html_generator_tool],
    system_prompt=make_system_prompt(
        "당신은 HTML 파일을 생성하는 전문가입니다. 웹 기반 자료조사를 하는 에이전트와 함께 일하고 있습니다. "
        "데이터나 차트를 HTML/CSS/JavaScript를 사용해 시각화하고 파일로 저장할 수 있습니다. "
    ),
)

def html_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    """
    HTML 시각화 작업을 수행하고, 완료 시 END로 가거나 추가 조사 필요 시 researcher로 바통을 넘깁니다.
    """
    # [ 1 ] HTML 시각화 에이전트를 실행합니다.
    result = html_agent.invoke(state) # [ 1 ]

    # [ 2 ] 마지막 응답의 발신자 이름을 'html_generator'로 기록합니다.
    result["messages"][-1] = HumanMessage( # [ 2 ]
        content=result["messages"][-1].content, name="html_generator"
    )
    last_message = result["messages"][-1]
    # [ 3 ] '최종 답변'이 포함되어 있으면 작업 종료(END), 아니면 다시 자료 조사자(researcher)에게 이동합니다.
    if "최종 답변" in last_message.content: # [ 3 ]
        goto = END
    else:
        goto = "researcher"

    # [ 4 ] 상태 업데이트와 다음 행선지(goto)를 Command 객체로 묶어 반환합니다.
    return Command( # [ 4 ]
        update={"messages": result["messages"]},
        goto=goto,
    )
