# ==============================================================================
# 📘 [CHAP10] LangGraph 에이전트 단독 테스트 스크립트
# ==============================================================================
# 💡 A2A 서버를 띄우기 전에, LangGraphAgent 본체가 계산과 시간 조회를 잘 수행하는지
#    직접 호출하여 점검(유닛 테스트)하는 스크립트입니다.
# ==============================================================================

import asyncio
import sys
from agent import LangGraphAgent


async def main():
    # LangGraph 에이전트 인스턴스 생성
    agent = LangGraphAgent()

    # 테스트할 질문 리스트 (계산 & 현재 시각)
    test_queries = [
        "25 곱하기 4는?",
        "현재 시간이 몇 시인가요?",
        "100을 3으로 나누면 얼마인가요?",
    ]
    for i, query in enumerate(test_queries, 1):
        print(f"\n[질문 {i}] {query}")
        print("-" * 60)

        try:
            # 에이전트 스트리밍 실행 결과 확인
            async for response in agent.stream(query):
                content = response.get("content", "")
                is_complete = response.get("is_task_complete", False)

                if content:
                    print(content)

                if is_complete:
                    print("-" * 60)
                    break

        except Exception as e:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
