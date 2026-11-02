# ============================================================================
# [9장 실습 1] 나만의 첫 번째 MCP 서버 만들기 (개인 비서 & 일기장 서버)
# 💡 일상 비유: MCP(Model Context Protocol)는 'AI계의 USB-C 통합 포트'입니다!
# 예전에는 마우스, 키보드, 모니터마다 연결 선 모양이 다 달랐지만 USB-C 하나로 통일되었듯,
# MCP 규격으로 도구(@mcp.tool)와 프롬프트(@mcp.prompt)를 만들어 두면 어떤 AI 모델이든 꽂아서 바로 쓸 수 있습니다.
# ============================================================================
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from datetime import datetime
from pathlib import Path


# [ 1 ] FastMCP를 사용해 "MCPServer"라는 이름의 MCP 서버(도구 상자)를 생성합니다.
mcp = FastMCP("MCPServer") # [ 1 ]


# [ 2 ] @mcp.tool() 데코레이터를 붙이면 일반 파이썬 함수가 AI가 꺼내 쓸 수 있는 'MCP 표준 도구'로 등록됩니다.
# 중요: 함수 아래의 독스트링("""설명""")은 AI가 "이 도구를 언제 써야 하지?"를 판단하는 사용 설명서 역할을 합니다!
@mcp.tool() # [ 2 ]
def read_user_info() -> str:
    """
    사용자의 정보를 담고 있는 txt 파일을 읽어옵니다.

    Returns:
        str: my_info.txt 파일의 내용
    """
    # [ 3 ] 현재 server.py 파일이 있는 폴더에서 my_info.txt 파일 경로를 찾습니다.
    info_file_path = Path(__file__).parent / "my_info.txt" # [ 3 ]

    try:
        # 파일을 UTF-8 인코딩으로 열어서 텍스트 내용을 모두 읽어 반환합니다.
        with open(info_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "my_info.txt 파일을 찾을 수 없습니다."
    except Exception as e:
        return f"파일을 읽는 중 오류가 발생했습니다: {str(e)}"

# 두 번째 MCP 도구: 대화 요약, 감정, 다짐을 받아 일기 파일(.txt)로 저장하는 도구입니다.
@mcp.tool()
def save_diary(conversation_summary: str, emotion: str, resolution: str) -> dict:
    """
    일기 형태로 텍스트 파일을 저장합니다. 모든 내용은 한국어로 작성합니다.

    Args:
        conversation_summary (str): 오늘의 대화 내용
        emotion (str): 오늘의 감정
        resolution (str): 오늘의 다짐

    Returns:
        dict: 저장 결과 (저장 경로, 파일명, 성공 여부)
    """ # [ 1 ]
    try:
        # [ 2 ] 일기를 모아둘 'diary' 폴더 경로를 지정하고, 폴더가 없으면 새로 만듭니다(exist_ok=True).
        diary_dir = Path(__file__).parent / "diary" # [ 2 ]
        diary_dir.mkdir(exist_ok=True)

        # [ 3 ] 파일명이 겹치지 않도록 현재 날짜와 시간(예: diary_20261102_143000.txt)으로 이름을 자동 생성합니다.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # [ 3 ]
        filename = f"diary_{timestamp}.txt"
        file_path = diary_dir / filename

        # [ 4 ] 일기 본문 맨 위에 적을 읽기 편한 한국어 날짜/시간 문자열을 만듭니다.
        timestamp_str = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분") # [ 4 ]

        # 일기장 양식에 맞춰 내용을 예쁘게 구성합니다.
        full_content = f"""[{timestamp_str}]

📝 오늘의 대화 내용
{conversation_summary}

💭 오늘의 감정
{emotion}

✨ 오늘의 다짐
{resolution}
"""

        # [ 5 ] 완성된 일기 내용을 실제 파일에 기록(write)합니다.
        with open(file_path, "w", encoding="utf-8") as f: # [ 5 ]
            f.write(full_content)

        return {
            "success": True,
            "path": str(file_path.absolute()),
            "filename": filename,
            "message": f"일기가 성공적으로 저장되었습니다."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"파일 저장 중 오류가 발생했습니다: {str(e)}"
        }

# @mcp.prompt() 데코레이터: 자주 쓰는 '대화 시작 템플릿(역할 부여 + 첫 인사)'을 서버에 미리 저장해 두는 기능입니다.
# 비유하자면 상담원에게 미리 쥐여주는 '친절 응대 매뉴얼 대본'입니다!
@mcp.prompt()
def default_prompt(message: str) -> list[base.Message]:
    return [
        base.AssistantMessage(
            "당신은 따뜻하고 공감 능력이 뛰어난 개인 비서입니다. \n"
            "- 사용자가 오늘 있었던 일을 편하게 이야기할 수 있도록 경청하고 공감해주세요\n"
            "- 대화가 자연스럽게 이어지도록 적절한 질문을 던져주세요\n"
            "- 사용자의 감정을 이해하고 긍정적인 피드백을 제공해주세요\n"
            "- 사용자가 대화를 끝내려 하면, 지금까지의 대화 내용을 담아 일기를 작성하세요\n"
            "- 존댓말을 사용하되, 친근하고 편안한 말투를 사용하세요"
        ),
        base.UserMessage(message),
    ]

# 이 파일을 직접 실행하면 표준 입출력(stdio: 키보드 입력/화면 출력 통로) 방식으로 MCP 서버가 가동됩니다.
if __name__ == "__main__":
    mcp.run(transport="stdio")
