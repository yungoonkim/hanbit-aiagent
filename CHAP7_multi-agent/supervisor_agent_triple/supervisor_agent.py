# ============================================================================
# [7.5 핸드오프 도구 기반 3중 슈퍼바이저] 3개 부서를 총괄하는 슈퍼바이저 (supervisor_agent.py)
# 비유: 3개의 직통 내선 전화 버튼(transfer_to_web_search, transfer_to_db_search, transfer_to_faq)을
#       책상 위에 두고, 질문 성격에 맞춰 한 부서 또는 여러 부서에 동시에 업무를 배정하는 본부장입니다.
# ============================================================================
from langchain.agents import create_agent

from handoff_tools import create_handoff_tool
from web_agent import create_web_agent
from db_agent import create_db_search_agent
from faq_agent import create_faq_agent
from settings import get_model, get_system_prompt

# [ 1 ] 그래프에 등록할 3명의 전문 에이전트 노드 사전을 정의합니다.
agents = { # [ 1 ]
    "web_search": create_web_agent,
    "db_search": create_db_search_agent,
    "faq": create_faq_agent
}

# [ 2 ] 슈퍼바이저가 각 하위 에이전트에게 작업을 넘길 때 사용할 핸드오프 도구 3개를 생성합니다.
handoff_tools = [ # [ 2 ]
    create_handoff_tool(
        agent_name="web_search",
        description="웹에서 최신 정보를 검색해야 할 때 사용합니다."
    ),
    create_handoff_tool(
        agent_name="db_search",
        description="내부 데이터베이스에서 문서 정보를 검색해야 할 때 사용합니다."
    ),
    create_handoff_tool(
        agent_name="faq",
        description="자주 묻는 질문에 대해 답변해야 할 때 사용합니다."
    )
]

# [ 3 ] Gemini 3.8 Flash 모델과 핸드오프 도구들을 결합해 슈퍼바이저 에이전트를 완성합니다.
model = get_model(model_name="gemini-3.8-flash") # [ 3 ]

supervisor = create_agent(
    model=model,
    tools=handoff_tools,
    system_prompt=get_system_prompt()
)
