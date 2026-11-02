# ============================================================================
# [6.4절] create_agent와 미들웨어 2: @before_model & @dynamic_prompt
#         (create_agent/middleware_with_node.py)
# ============================================================================
# 이 파일은 두 가지 다른 성격의 미들웨어를 한 에이전트에 동시에 적용해 비교하는 실습입니다.
#
# 💡 비전공자를 위한 일상 비유:
# 1. `@before_model` (검문소 부스 추가 O):
#    - 건물 입구에 아예 **독립된 보안 검색대 방(노드)**을 하나 새로 짓는 것과 같습니다.
#    - 금지어("바보", "멍청이")가 있으면 AI 방에 들어가기도 전에 출입을 차단합니다!
# 2. `@dynamic_prompt` (별도 부스 추가 X):
#    - 새로운 방을 짓지 않고, AI가 대답하기 직전에 귀에 대고
#      "이번 손님에게는 존댓말(또는 반말)로 대답해!"라고 **귓속말(시스템 프롬프트)**만 살짝 넣어주는 방식입니다.
# ============================================================================

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, dynamic_prompt, AgentState, ModelRequest
from langgraph.runtime import Runtime

from tools import tools

# .env 파일에서 API 키를 불러옵니다.
load_dotenv()


# ===== 모델 정의 =====
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# 금지어 목록 (보안 검색대에서 적발할 나쁜 말 리스트)
BLOCKED_WORDS = ["바보", "멍청이", "나쁜말"]


# ===== 미들웨어 정의 (노드가 추가되는 방식) =====
# @before_model 데코레이터를 붙이면 그래프 설계도에 'content_filter_middleware'라는 실제 노드가 추가됩니다!
@before_model
def content_filter_middleware(state: AgentState, runtime: Runtime): # [ 1 ]
    """
    금지어를 필터링하는 미들웨어
    - 그래프에 'content_filter_middleware' 노드가 추가됨
    - 금지어 감지 시 예외 발생으로 중단
    """
    # 마지막 메시지 확인 (방금 사용자가 입력한 문장)
    if state["messages"]:
        last_msg = state["messages"][-1]
        content = getattr(last_msg, 'content', str(last_msg))

        # 금지어 검사: 금지어가 하나라도 포함되어 있으면 즉시 에러(ValueError)를 일으켜 멈춥니다.
        for word in BLOCKED_WORDS:
            if word in content:
                print(f"[before_model] 🚫 금지어 감지: '{word}'")
                raise ValueError(f"부적절한 표현이 감지되었습니다: '{word}'")

        print(f"[before_model] ✅ 입력 검증 통과")

    return None  # 문제가 없으면 아무것도 수정하지 않고(None) 다음 단계(model 노드)로 통과시킵니다.


# ===== 미들웨어 정의 2: @dynamic_prompt (동적 시스템 프롬프트) =====
# @dynamic_prompt는 내부적으로 @wrap_model_call 위에서 동작하므로 그래프에 별도 노드가 생기지 않습니다.
@dynamic_prompt
def random_tone_prompt(request: ModelRequest) -> str: # [ 2 ]
    """
    랜덤하게 말투를 변경하는 미들웨어
    - 존댓말 또는 반말 프롬프트를 랜덤 선택
    - @wrap_model_call 기반이므로 노드 추가 X
    """
    import random

    # 동전 던지기처럼 50% 확률로 존댓말/반말 시스템 프롬프트를 반환합니다.
    if random.choice([True, False]):
        print(f"[dynamic_prompt] 존댓말 모드")
        return "당신은 친절한 AI입니다. 항상 존댓말로 정중하게 답변하세요."
    else:
        print(f"[dynamic_prompt] 반말 모드")
        return "너는 친근한 AI야. 항상 반말로 편하게 답변해."


# ===== 에이전트 생성 =====
# 두 미들웨어를 리스트로 함께 등록합니다.
agent = create_agent( # [ 3 ]
    model=model,
    tools=tools,
    middleware=[
        content_filter_middleware,  # @before_model: 노드 추가 O, 금지어 필터링
        random_tone_prompt,         # @dynamic_prompt: 노드 추가 X (wrap_model_call 기반)
    ]
)

if __name__ == "__main__":
    from pathlib import Path

    # 그래프 이미지를 그려보면 'content_filter_middleware.before_model' 노드가 추가된 것을 볼 수 있습니다!
    save_path = Path(__file__).parent / "middleware_with_node.png"
    graph_image = agent.get_graph().draw_mermaid_png()

    with open(save_path, "wb") as f:
        f.write(graph_image)

    # 테스트 1: 정상 입력 -> 검문소 통과 후 계산기 도구 실행 및 응답
    print("=" * 50)
    print("테스트 1: 정상 입력")
    print("=" * 50)
    response = agent.stream({"messages": ["15와 7을 더해주세요."]})
    for chunk in response:
        for node, value in chunk.items():
            if node:
                print(f"\n--- {node} ---")
            if value and "messages" in value:
                print(value['messages'][0].content)

    # 테스트 2: 금지어 포함 입력 -> AI 모델을 부르기도 전에 검문소에서 즉시 차단!
    print("\n" + "=" * 50)
    print("테스트 2: 금지어 포함 입력")
    print("=" * 50)
    try:
        response = agent.stream({"messages": ["바보야 10과 5를 더해줘"]})
        for chunk in response:
            for node, value in chunk.items():
                if node:
                    print(f"\n--- {node} ---")
                if value and "messages" in value:
                    print(value['messages'][0].content)
    except ValueError as e:
        print(f"❌ 차단됨: {e}")
