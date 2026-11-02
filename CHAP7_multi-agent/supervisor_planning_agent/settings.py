# ============================================================================
# [7.6 기획(Planning) 기반 슈퍼바이저] 공통 상태 및 모델 설정 (settings.py)
# 비유: 큰 프로젝트를 진행할 때 필요한 '전체 체크리스트(plan)'와 '완료된 업무 일지(past_steps)'를
#       기록하는 공용 프로젝트 보드(State)를 정의합니다.
# ============================================================================
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState

def get_model(model_name: str = "gemini-3.8-flash"):
    """기획자, 슈퍼바이저, 리서처, 작가 에이전트가 공통으로 사용할 Gemini 3.8 Flash 모델을 반환합니다."""
    llm = ChatGoogleGenerativeAI(model=model_name)
    return llm

class State(MessagesState):
    """전체 그래프에서 공유하는 상태: 대화 기록(messages), 남은 계획(plan), 완료된 단계 기록(past_steps)"""
    plan: List[str]
    past_steps: List[tuple]

class SupervisorState(MessagesState):
    """슈퍼바이저 노드가 사용하는 상태: 다음 호출할 에이전트(next) 정보가 추가됩니다."""
    plan: List[str]
    next: str
    past_steps: List[tuple]
