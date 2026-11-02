# ==============================================================================
# 📘 [CHAP10] A2A Hello World - 에이전트 실행자 (AgentExecutor)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - HelloWorldAgent = 실제로 요리(인사말 만들기)를 담당하는 '주방 요리사'
# - HelloWorldAgentExecutor = 손님의 주문(RequestContext)을 받아 요리사에게 전달하고,
#   완성된 요리를 배식구(EventQueue)에 올려놓는 '홀 매니저(실행 총괄자)'입니다!
# ==============================================================================

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message


class HelloWorldAgent:
    """
    간단한 Hello World 에이전트 (실제 비즈니스 로직을 수행하는 주방 요리사)

    이 에이전트는 외부 도구나 LLM 없이 항상 반가운 "Hello World!" 인사말을 반환합니다.
    A2A 프로토콜의 뼈대(통신 구조)를 가장 쉽게 이해하기 위한 입문용 에이전트입니다.
    """

    async def invoke(self) -> str:
        """
        에이전트 실행 메서드 (요리사가 인사말을 만들어 반환하는 함수)

        Returns:
            str: "Hello World!" 인사 메시지
        """
        return "Hello World! 안녕하세요, A2A 에이전트입니다!"


class HelloWorldAgentExecutor(AgentExecutor):
    """
    A2A 프로토콜 규격에 맞춰 에이전트를 실행하는 실행자 (홀 매니저)

    A2A 서버(A2AStarletteApplication)는 에이전트 내부 코드가 어떻게 생겼는지 몰라도,
    이 AgentExecutor 표준 인터페이스(execute, cancel)만 있으면 어떤 에이전트와도 대화할 수 있습니다.
    """

    def __init__(self):
        """에이전트 인스턴스 초기화 (주방에 요리사 배치하기)"""
        self.agent = HelloWorldAgent()

    async def execute( # [ 1 ]
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        [핵심] 클라이언트 요청이 들어왔을 때 에이전트를 실행하고 결과를 이벤트 큐에 전송합니다.

        Args:
            context: 요청 컨텍스트 (손님이 보낸 주문서: 사용자 메시지, 세션 정보 등)
            event_queue: 결과를 전송할 이벤트 큐 (완성된 요리를 내보내는 배식 컨베이어 벨트)
        """
        # 1. 주방 요리사(HelloWorldAgent)를 호출하여 인사말 결과를 받아옵니다.
        result = await self.agent.invoke() # [ 2 ]

        # 2. 결과 문자열을 A2A 표준 텍스트 메시지(new_agent_text_message)로 예쁘게 포장한 뒤,
        #    이벤트 큐(배식구)에 넣어서 클라이언트에게 전달합니다.
        await event_queue.enqueue_event(new_agent_text_message(result)) # [ 3 ]

    async def cancel( # [ 4 ]
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        작업 취소 메서드 (손님이 주문 취소를 요청했을 때 호출됨)

        이 간단한 Hello World 예제에서는 작업이 즉시 끝나므로 취소 기능을 별도로 지원하지 않습니다.
        """
        raise Exception("cancel은 지원되지 않습니다")
