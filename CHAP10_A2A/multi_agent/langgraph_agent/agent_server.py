# ==============================================================================
# 📘 [CHAP10] LangGraph 에이전트 A2A 서버 (10001번 포트)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - 이 서버는 '수학 계산'과 '시간 정보 조회' 두 가지 자격증(AgentSkill)을 명함(AgentCard)에
#   적어두고 10001번 포트 사무실에서 대기하는 전문 에이전트 서버입니다.
# ==============================================================================

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executor import LangGraphAgentExecutor


def create_agent_card() -> AgentCard: # [ 1 ]
    """
    LangGraph 에이전트의 명함(AgentCard) 생성 함수
    - 보유 스킬 1: 수학 계산 (langgraph_calculator)
    - 보유 스킬 2: 시간 정보 조회 (langgraph_time_info)
    """
    math_skill = AgentSkill(
        id='langgraph_calculator',
        name='수학 계산',
        description='수학 연산을 수행합니다 (덧셈, 뺄셈, 곱셈, 나눗셈 등)',
        tags=['계산', '수학', 'math', 'calculator'],
        examples=['2 + 2는 얼마인가요?', '10 * 5 계산해줘', '100 나누기 4는?'],
    )

    time_skill = AgentSkill(
        id='langgraph_time_info',
        name='시간 정보 조회',
        description='현재 날짜와 시간 정보를 제공합니다',
        tags=['시간', '날짜', 'datetime', 'time'],
        examples=['현재 시간은?', '오늘 날짜는?', '지금 몇 시야?'],
    )

    # 실시간 스트리밍(streaming=True) 지원 명시
    capabilities = AgentCapabilities( # [ 2 ]
        streaming=True,
        input_modes=['text'],
        output_modes=['text'],
    )

    agent_card = AgentCard( # [ 3 ]
        name='LangGraph Agent',
        description='수학 계산과 시간 정보를 제공하는 다목적 에이전트',
        url='http://localhost:10001',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=capabilities,
        skills=[math_skill, time_skill],
    )

    return agent_card


def main():
    """LangGraph A2A 에이전트 서버를 10001번 포트에서 시작합니다."""
    # 에이전트 카드(명함) 생성
    agent_card = create_agent_card() # [ 4 ]

    # 에이전트 실행자 생성
    agent_executor = LangGraphAgentExecutor() # [ 5 ]

    # 작업 저장소 (메모리 기반)
    task_store = InMemoryTaskStore() # [ 6 ]

    # 요청 처리기 생성
    request_handler = DefaultRequestHandler( # [ 7 ]
        agent_executor=agent_executor,
        task_store=task_store,
    )

    # A2A 서버 애플리케이션 생성
    server_app = A2AStarletteApplication( # [ 8 ]
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # 10001번 포트에서 서버 구동
    uvicorn.run( # [ 9 ]
        server_app.build(),
        host='0.0.0.0',
        port=10001,
    )


if __name__ == '__main__':
    main()
