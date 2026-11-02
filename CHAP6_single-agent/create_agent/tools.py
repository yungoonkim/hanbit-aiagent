# ============================================================================
# [6.4절] 미들웨어 실습용 사칙연산 계산기 도구 (create_agent/tools.py)
# ============================================================================
# 이 파일은 6.4절 `create_agent`와 미들웨어(Middleware) 실습에서
# 에이전트가 사용할 간단한 사칙연산(더하기, 빼기, 곱하기, 나누기) 계산기 도구입니다.
#
# 💡 비전공자를 위한 일상 비유:
# - AI(언어 모델)는 글을 쓰는 능력은 뛰어나지만, 암산으로 복잡한 계산을 할 때는 실수할 수 있습니다.
# - 그래서 AI의 손에 정확한 '전자계산기(calculator 도구)'를 들려주어 계산 실수를 방지합니다!
# ============================================================================

from langchain.tools import tool


@tool
def calculator(a: int, b: int, operation: str) -> str:
    """
    간단한 계산기 도구입니다.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        operation: 연산 종류 (add, subtract, multiply, divide)
    """
    # AI가 전달한 연산 종류(operation)에 따라 알맞은 사칙연산을 수행합니다.
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        # 0으로 나누면 에러가 나므로 친절한 안내 문구를 반환합니다.
        result = a / b if b != 0 else "0으로 나눌 수 없습니다"
    else:
        return f"지원하지 않는 연산: {operation}"

    # 계산 식과 결과를 문자열로 깔끔하게 정리해 AI에게 돌려줍니다.
    return f"{a} {operation} {b} = {result}"


# 에이전트에게 전달할 도구 목록
tools = [calculator]
