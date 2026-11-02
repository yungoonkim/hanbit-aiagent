# ============================================================================
# [6.5절] Self-RAG 품질 심사관 (조건부 엣지 함수 모음 - rag_agent/edges.py)
# ============================================================================
# 이 파일은 일반 RAG의 고질적인 문제인 "엉뚱한 문서 검색"과 "AI의 거짓말(환각, Hallucination)"을
# 막기 위해 두 명의 AI 심사관(Grader)이 O/X('yes'/'no')로 엄격히 판정하는 갈림길 함수들입니다.
#
# 💡 비전공자를 위한 일상 비유:
# 1. `decide_to_generate` (1차 심사관 - 자료 적합성 검사):
#    - 사서가 가져온 책 페이지가 손님의 질문과 진짜 관련이 있는지 확인합니다.
#    - 관련이 없으면('no') -> "질문을 다시 다듬어서 새로 찾아와!(transform_query)"라고 돌려보냅니다.
# 2. `check_hallucinations` (2차 심사관 - 팩트 체크 / 환각 검증):
#    - 집필가가 쓴 최종 답변이 찾아온 문서(팩트)에 기반한 내용인지 검사합니다.
#    - 문서에 없는 내용을 지어냈다면('no') -> "다시 똑바로 써!(generate)"라고 재작성을 지시합니다.
# ============================================================================

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 심사관 역할을 수행할 Gemini 3.8 Flash 모델입니다.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


# ===== [1차 심사표 양식] 문서 관련성 채점표 =====
class Grade(BaseModel):
    """관련성 확인을 위한 점수 스키마"""
    # AI가 장황하게 말하지 않고 오직 'yes' 또는 'no'로만 딱 떨어지게 답하도록 강제하는 양식입니다.
    binary_score: str = Field(description="문서가 질문과 관련이 있는지 여부, 'yes' 또는 'no'")


def decide_to_generate(state):
    """
    답변을 생성할지, 아니면 질문을 다시 생성할지 결정합니다.
    """

    print("----- ASSESS GRADED DOCUMENTS -----")
    # [ 1 ] 이미 3번 이상 재검색(retry_num >= 3)을 시도했다면, 무한 루프에 빠지지 않도록 무조건 generate로 보냅니다.
    if state.get("retry_num", 0) >= 3: # [ 1 ]
        return "generate"

    # [ 2 ] with_structured_output(Grade): AI가 반드시 Grade 채점표 양식(binary_score='yes'/'no')에 맞춰 답하도록 만듭니다!
    grader = llm.with_structured_output(Grade) # [ 2 ]

    # [ 3 ] 1차 심사관에게 줄 채점 기준표(프롬프트)입니다.
    grader_prompt = ChatPromptTemplate.from_messages( # [ 3 ]
        [
            (
                "system",
                """
                당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 평가자입니다.
                문서가 사용자 질문과 관련된 키워드나 의미를 포함하고 있다면 관련성이 있다고 평가하세요.
                엄격한 테스트일 필요는 없습니다. 목표는 잘못된 검색 결과를 필터링하는 것입니다.
                문서가 질문과 관련이 있는지를 나타내는 'yes' 또는 'no'의 이진 점수를 제공하세요.
                """
            ),
            (
                "user",
                "검색된 문서: {context} \n\n 사용자 질문: {question} \n\n 관련성 점수:"
            ),
        ]
    )

    chain = grader_prompt | grader

    question = state.get("question", "")
    context = state.get("context", "")

    if not context or not question:
        print("---ERROR: Missing context or question, defaulting to generate---")
        return "generate"

    # 심사를 실행하고 채점표에서 'yes' 또는 'no' 점수를 꺼냅니다.
    score = chain.invoke({"question": question, "context": context})
    grade = score.binary_score # [ 4 ]

    # 관련이 없다고 판정('no')되면 질문을 교정하는 노드로 이동하고, 관련이 있으면('yes') 답변 생성 노드로 이동합니다.
    if grade == "no":
        print(
            "---DECISION: RETRIEVED DOCUMENT ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"



# ===== [2차 심사표 양식] 환각(거짓말) 여부 채점표 =====
class GradeHallucinations(BaseModel):
    """생성된 답변의 환각 여부를 판단하기 위한 점수 스키마"""

    binary_score: str = Field(
        description="답변이 사실에 근거하고 있는지 여부, 'yes' 또는 'no'"
    )

def check_hallucinations(state):
    """
    생성된 답변이 문서에 근거하고 질문에 답하는지 판단합니다.
    """

    print("----- CHECK HALLUCINATIONS -----")
    question = state["question"]
    context = state["context"]
    answer = state["answer"]

    # 2차 심사관 역시 GradeHallucinations 채점표('yes' 또는 'no')로만 대답하도록 설정합니다.
    structured_llm = llm.with_structured_output(GradeHallucinations)

    system = """당신은 LLM이 생성한 답변이 검색된 사실들에 근거하고 있는지 평가하는 평가자입니다.
    'yes' 또는 'no'의 이진 점수를 제공하세요. 'yes'는 답변이 사실들에 근거하고 있음을 의미합니다."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", "질문: {question} \n\n 사실 집합: \n\n {context} \n\n LLM 생성 답변: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm

    score = hallucination_grader.invoke(
        {"question": question, "context": context, "generation": answer}
    )
    grade = score.binary_score

    # 답변이 문서 사실에 잘 근거하고 있다면('yes') 통과("support" -> 종료),
    # 근거 없이 지어낸 말이라면('no') 불합격("not supported" -> 다시 답변 생성)을 반환합니다!
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "support"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
