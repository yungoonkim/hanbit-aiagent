# ============================================================================
# [6.3절] 코딩 에이전트용 사용자 정의 도구 모음 (coding_agent/tools.py)
# ============================================================================
# 이 파일은 AI가 직접 파이썬 코드를 실행해 보고(python_exec_tool),
# 검증된 코드를 실제 파일로 저장할 수 있게 해주는(file_write_tool) 도구 상자입니다.
#
# 💡 비전공자를 위한 일상 비유:
# - @tool 데코레이터는 평범한 파이썬 함수에 "AI 전용 사용 설명서 스티커"를 붙여서,
#   AI가 "아! 이 도구는 언제, 어떤 입력값을 넣어서 쓰면 되는구나!"라고 이해하게 만듭니다.
# - 특히 함수 아래에 적는 큰따옴표 3개("""...""") 설명글(Docstring)은
#   사람만 보는 메모가 아니라 **AI가 도구를 고를 때 읽는 핵심 안내서**입니다!
# ============================================================================

from pydantic import Field
from langchain.tools import tool


# [도구 1] 파이썬 코드 실행기 (AI의 연습장 & 실험실)
@tool
def python_exec_tool(
    imports: str = Field(description="임포트 구문"), # [ 1 ]
    code: str = Field(description="임포트 구문을 제외한 코드 블록"),
) -> str:
    """
    파이썬 코드를 실행하는 도구입니다. 만약 코드 실행에 실패하면 에러 메시지를 반환합니다.
    실행 결과를 확인하고 싶다면 `print(...)`를 사용하여 출력해야 합니다.

    Args:
        imports: 임포트 구문
        code: 임포트 구문을 제외한 코드 블록

    Returns:
        실행 결과 또는 에러 메시지
    """ # [ 2 ]
    # [ 3 ] Check imports: 먼저 필요한 부품(라이브러리)이 잘 불러와지는지 테스트합니다.
    # 비유: 요리를 시작하기 전에 필요한 식재료(모듈)가 주방에 있는지 먼저 점검하는 단계입니다.
    try:
        exec(imports)
    except Exception as e:
        return f"모듈을 임포트하는 데 실패했습니다. ERROR: {repr(e)}"


    # [ 4 ] Check execution: 식재료 점검이 끝났으면, 실제 요리(코드 본문)를 실행해 봅니다.
    # 만약 코드에 오타나 버그가 있다면 에러 메시지를 돌려주어 AI가 스스로 코드를 고치게 합니다.
    try:
        exec(imports + "\n" + code)
    except Exception as e:
        return f"코드 실행에 실패했습니다. ERROR: {repr(e)}"

    # 에러 없이 무사히 실행되었다면 성공 메시지를 반환합니다.
    result_str = f"성공적으로 코드가 실행되었습니다. :\n```python\n{code}\n```"

    return result_str


# [도구 2] 파일 저장기 (AI의 문서 보관함)
@tool
def file_write_tool(
    file_path: str = Field(description="생성/수정할 파일의 경로"),
    content: str = Field(description="파일에 작성할 내용")
) -> str:
    """
    파일을 생성하거나 내용을 작성하는 도구입니다.

    Args:
        file_path: 생성/수정할 파일의 경로
        content: 파일에 작성할 내용

    Returns:
        성공/실패 메시지
    """
    # 비유: 연습장(python_exec_tool)에서 실행이 잘 되는 것을 확인한 뒤,
    # 깨끗한 새 공책(파일)을 꺼내 완성된 코드를 정식으로 적어 저장하는 단계입니다.
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"파일 '{file_path}'에 성공적으로 작성했습니다."
    except Exception as e:
        return f"파일 작성 실패: {repr(e)}"
