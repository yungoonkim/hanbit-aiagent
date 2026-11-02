# ==============================================================================
# 📘 [CHAP10] A2A Hello World - 테스트 클라이언트 (TestClient)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# 1. A2ACardResolver = 상대방 에이전트 사무실에 방문해 '명함(AgentCard) 먼저 주세요!'라고 요청하는 역할
# 2. A2AClient = 받은 명함에 적힌 주소와 양식에 맞춰 '외주 업무 요청서(SendMessageRequest)'를 보내는 연락 담당자
# ==============================================================================

import asyncio
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    SendMessageRequest,
    TextPart,
)


async def main():
    """
    메인 함수: A2A 에이전트 서버와 통신하는 클라이언트 예제

    이 함수는:
    1. 에이전트 서버(9999번 포트)에서 에이전트 카드(명함)를 가져와 정보를 확인합니다.
    2. 명함 정보를 바탕으로 A2A 클라이언트를 생성합니다.
    3. 에이전트에게 인사 메시지를 전송합니다.
    4. 에이전트가 돌려준 응답을 화면에 출력합니다.
    """
    # 접속할 Hello World 에이전트 서버 주소
    agent_url = 'http://localhost:9999' # [ 1 ]

    # 비동기 HTTP 통신 클라이언트(httpx.AsyncClient)를 열고 대화를 시작합니다.
    async with httpx.AsyncClient() as httpx_client:
        # 1단계: 명함 수집기(A2ACardResolver)를 통해 상대 에이전트의 명함(AgentCard) 조회
        # 내부적으로 `http://localhost:9999/.well-known/agent-card.json`에 접속해 정보를 가져옵니다.
        card_resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=agent_url,
        )
        agent_card = await card_resolver.get_agent_card() # [ 2 ]

        # 가져온 명함(AgentCard) 내용을 화면에 보기 좋게 출력합니다.
        print(f"\n에이전트 정보:")
        print(f"   - 이름: {agent_card.name}")
        print(f"   - 설명: {agent_card.description}")
        print(f"   - 버전: {agent_card.version}")
        print(f"   - 스킬 개수: {len(agent_card.skills)}")

        if agent_card.skills:
            print(f"\n   스킬 목록:")
            for skill in agent_card.skills:
                print(f"   • {skill.name}: {skill.description}")
                if skill.examples:
                    print(f"     예시: {', '.join(skill.examples[:3])}")

        # 2단계: 확인한 명함(agent_card)을 넣어 A2A 전용 통신기(A2AClient) 생성
        client = A2AClient( # [ 3 ]
            httpx_client=httpx_client,
            agent_card=agent_card,
        )

        # 테스트로 보내볼 인사말 목록
        test_messages = [ # [ 4 ]
            "안녕하세요!",
            "Hello",
            "무엇을 할 수 있나요?",
        ]

        # 3단계: 준비한 메시지를 하나씩 에이전트에게 보내고 답장을 받습니다.
        for i, user_input in enumerate(test_messages, 1):
            print(f"\n[메시지 {i}/{len(test_messages)}]")
            print(f"👤 사용자: {user_input}")

            # A2A 표준 메시지(편지 알맹이) 생성: 고유 ID(uuid4)와 텍스트 조각(TextPart)을 담습니다.
            message = Message( # [ 5 ]
                role='user',
                parts=[TextPart(kind='text', text=user_input)],
                message_id=uuid4().hex,
            )

            # A2A 메시지 전송 요청서(편지 봉투) 생성
            request = SendMessageRequest( # [ 6 ]
                id=uuid4().hex,
                params=MessageSendParams(message=message),
            )

            # 4단계: 에이전트에게 요청서를 발송하고 응답을 기다립니다!
            response = await client.send_message(request) # [ 7 ]

            # 응답 상자(response.root.result.parts)를 열어서 에이전트가 보낸 텍스트를 꺼냅니다.
            if hasattr(response, 'root') and response.root:
                result = response.root.result

                if hasattr(result, 'parts') and result.parts:
                    for part in result.parts:
                        if hasattr(part, 'root'):
                            text_part = part.root
                            if hasattr(text_part, 'text'):
                                print(f"🤖 에이전트: {text_part.text}")

            if i < len(test_messages):
                await asyncio.sleep(0.5)


if __name__ == '__main__':
    asyncio.run(main())
