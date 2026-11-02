# ==============================================================================
# 📘 [CHAP10] A2A Hello World - 에이전트 서버 (AgentServer)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - AgentCard = 에이전트의 '공식 비즈니스 명함' (이름, 주소, 특기/스킬, 연락 방법이 적혀 있음)
#   다른 에이전트나 클라이언트가 `/.well-known/agent-card.json` 주소로 찾아오면 이 명함을 건네줍니다!
# - AgentSkill = 명함 뒷면에 적힌 '전문 분야 목록' ("저는 인사말하기를 잘합니다! 예시: 안녕, hello")
# - A2AStarletteApplication = 명함 배부처와 안내 데스크를 갖춘 '에이전트 매장 건물'
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
from agent_executor import HelloWorldAgentExecutor


def create_agent_card() -> AgentCard:
    """
    에이전트 카드(명함) 생성 함수

    AgentCard는 에이전트의 메타데이터(이름, 설명, URL, 보유 기술)를 담고 있으며,
    다른 에이전트(예: 오케스트레이터)나 클라이언트가 이 에이전트의 기능을 파악할 수 있게 합니다.

    Returns:
        AgentCard: 에이전트 정보 카드 (명함)
    """
    # 1. AgentSkill: 에이전트가 수행할 수 있는 구체적인 특기(스킬) 정의
    # 비유: 명함 뒷면에 "전문 업무: Hello World 인사 / 질문 예시: 안녕, hello"라고 적어두는 것
    skill = AgentSkill( # [ 1 ]
        id='hello_world',
        name='Hello World 인사',
        description='간단한 인사말을 반환합니다',
        tags=['인사', 'hello world', '기본'],
        examples=['안녕', '안녕하세요', 'hi', 'hello'],
    )

    # 2. AgentCapabilities: 에이전트의 기술적 통신 능력 정의
    # 비유: "우리 매장은 실시간 중계(streaming)는 안 되고, 글자(text)로만 주문받고 답변합니다"라는 안내
    capabilities = AgentCapabilities( # [ 2 ]
        streaming=False,  # 스트리밍(실시간 조각 전송) 지원 여부
        input_modes=['text'],  # 받을 수 있는 입력 형식
        output_modes=['text'],  # 내보낼 수 있는 출력 형식
    )

    # 3. AgentCard: 에이전트의 전체 프로필(명함) 완성
    agent_card = AgentCard( # [ 3 ]
        name='Hello World 에이전트',
        description='A2A 프로토콜을 학습하기 위한 가장 간단한 에이전트입니다',
        url='http://localhost:9999',  # 이 에이전트가 상주하는 서버 주소
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=capabilities,
        skills=[skill],
    )

    return agent_card


def main():
    """
    A2A 에이전트 서버 시작 함수 (매장 오픈 준비 및 영업 개시!)

    이 함수는 다음 5단계로 서버를 준비합니다:
    1. AgentCard를 생성하여 에이전트 명함을 준비합니다.
    2. HelloWorldAgentExecutor로 실제 일을 처리할 실행자(매니저)를 고용합니다.
    3. InMemoryTaskStore로 작업 진행 내역을 기록할 장부(메모리 저장소)를 준비합니다.
    4. DefaultRequestHandler로 손님 응대 매뉴얼(요청 처리기)을 구성합니다.
    5. A2AStarletteApplication으로 서버 앱을 조립하고 uvicorn으로 9999번 포트에서 문을 엽니다!
    """
    # 1. 에이전트 카드(명함) 생성
    agent_card = create_agent_card() # [ 1 ]

    # 2. 에이전트 실행자(홀 매니저) 생성
    agent_executor = HelloWorldAgentExecutor() # [ 2 ]

    # 3. 작업 저장소 생성 (진행 중인 업무 상태를 컴퓨터 메모리에 기록하는 장부)
    task_store = InMemoryTaskStore() # [ 3 ]

    # 4. 기본 요청 처리기 생성 (매니저와 장부를 연결)
    request_handler = DefaultRequestHandler( # [ 4 ]
        agent_executor=agent_executor,
        task_store=task_store,
    )

    # 5. A2A 표준 웹 서버 애플리케이션 조립 (명함과 요청 처리기 등록)
    server_app = A2AStarletteApplication( # [ 5 ]
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # 6. uvicorn 웹 서버 실행 (http://localhost:9999 에서 손님 맞이를 시작합니다!)
    uvicorn.run( # [ 6 ]
        server_app.build(),
        host='0.0.0.0',
        port=9999,
    )


if __name__ == '__main__':
    main()
