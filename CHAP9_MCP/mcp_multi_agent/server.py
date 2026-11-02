# ============================================================================
# [9장 실습 2] 멀티 에이전트용 로컬 파일 관리 MCP 서버 (FileSearch 서버)
# 💡 일상 비유: 사내 문서실 담당 직원이 쓸 '파일 목록 조회, 상세 정보 읽기, 파일 저장'이라는
# 3가지 전용 도구를 담아둔 MCP 서버입니다.
# ============================================================================
from mcp.server.fastmcp import FastMCP
import os
import stat
import time
from typing import List


# [ 1 ] "FileSearch"라는 이름의 MCP 서버를 생성하고, 서버의 역할을 설명(instructions)에 적어줍니다.
mcp = FastMCP( # [ 1 ]
    "FileSearch",
    instructions="로컬 파일 검색을 도와주는 어시스턴트입니다."
)


# [ 2 ] 도구 1 - 파일 목록 호출: 특정 폴더(directory) 안에 어떤 파일들이 있는지 목록을 가져옵니다.
@mcp.tool() # [ 2 ]
async def file_listup(directory: str) -> List[str]:
    """지정된 디렉토리 경로의 파일 이름 목록을 반환합니다."""
    try:
        return os.listdir(directory)
    except Exception as e:
        return [f"오류: {str(e)}"]


# [ 3 ] 도구 2 - 파일 상세 정보 및 본문 읽기: 파일의 크기, 생성/수정 시간, 권한, 텍스트 내용을 한꺼번에 조사합니다.
@mcp.tool() # [ 3 ]
async def file_info(path: str) -> dict:
    """
    지정된 경로의 파일 또는 디렉토리에 대한 상세 정보를 반환합니다.
    """
    if not os.path.exists(path):
        return {"error": f"경로 '{path}' 가 존재하지 않습니다."}

    try:
        # 운영체제(OS)로부터 파일의 상태 정보(크기, 시간, 권한 등)를 가져옵니다.
        stat_info = os.stat(path)

        # 파일 권한(예: -rw-r--r--)을 사람이 읽기 쉬운 문자열로 변환합니다.
        permissions = stat.filemode(stat_info.st_mode)

        # 파일 안의 실제 텍스트 내용도 함께 읽어옵니다.
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "path": os.path.abspath(path),
            "content": content,
            "type": "file",
            "size": stat_info.st_size,  # 파일 크기 (bytes 단위)
            "created_time": time.ctime(stat_info.st_birthtime),
            "modified_time": time.ctime(stat_info.st_mtime),
            "access_time": time.ctime(stat_info.st_atime),
            "permissions": permissions,
        }
    except Exception as e:
        return {"error": str(e)}


# [ 4 ] 도구 3 - 파일 쓰기 저장 담당: 정리된 내용을 새로운 마크다운/텍스트 파일로 저장합니다.
@mcp.tool() # [ 4 ]
async def save_file(content: str, output_path="file_info.md") -> str:
    """제공된 내용을 지정된 output_path의 텍스트 파일에 작성합니다."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


# [ 5 ] 표준 입출력(stdio) 통신 방식으로 FileSearch MCP 서버를 실행합니다.
if __name__ == "__main__": # [ 5 ]
    mcp.run(transport="stdio")
