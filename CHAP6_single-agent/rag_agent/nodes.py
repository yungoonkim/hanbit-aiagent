# ============================================================================
# [6.5절] Self-RAG / Corrective RAG 에이전트의 5대 작업 부서 (rag_agent/nodes.py)
# ============================================================================
# 이 파일은 RAG 에이전트 그래프를 구성하는 5개의 핵심 노드(작업자 함수)를 정의합니다.
#
# 💡 비전공자를 위한 5개 부서 역할 소개:
# 1. `chatbot` (안내 데스크): 손님 질문을 듣고 일상 대화면 바로 답하고, 맞춤법 질문이면 도서관 검색을 요청합니다.
# 2. `retrieve` (도서관 사서): 실제로 벡터 DB에서 질문과 관련된 문서 조각 3개를 찾아옵니다.
# 3. `context_organizer` (자료 정리원): 찾아온 문서의 지저분한 줄바꿈/공백을 깔끔하게 정돈합니다(페이지 번호 유지!).
# 4. `transform_query` (질문 교정 전문가): 검색 결과가 영 딴판일 때, 검색이 더 잘 되도록 질문을 똑똑하게 다시 씁니다.
# 5. `generate` (최종 집필가): 정리된 문서만 근거로 삼아 출처(페이지 번호)와 함께 답변을 작성합니다.
# ============================================================================

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage

from retriever import retriever, retriever_tool
from state import AgentState


# 모든 부서에서 공통으로 사용할 Gemini 3.8 Flash 모델을 준비합니다.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


# ===== [부서 1] 안내 데스크 노드 (chatbot) =====
def chatbot(state: AgentState):
    """
    검색(Retriever) 도구를 바인딩 한 LLM 모델에 현재 메시지 상태를 입력하여 응답을 생성합니다.
    질문이 주어지면 검색 도구를 도구호출 하거나 일반 답변하며 종료할지 결정할 수 있습니다.
    """
    print("----- [CHATBOT] -----")
    messages = state["messages"]
    # [ 1 ] AI에게 '한글 맞춤법 PDF 검색 도구(retriever_tool)' 사용 설명서를 쥐여줍니다.
    llm_with_tools = llm.bind_tools([retriever_tool]) # [ 1 ]
    # [ 2 ] 질문을 보고 스스로 판단해 일반 답변을 할지, pdf_search 도구를 호출할지 결정합니다.
    response = llm_with_tools.invoke(messages) # [ 2 ]

    # [ 3 ] AI의 응답과 함께 사용자의 원래 질문(messages[-1].content)을 question 칸에 기록해 둡니다.
    return { # [ 3 ]
        "messages": [response],
        "question" : messages[-1].content
    }


# ===== [부서 2] 도서관 사서 노드 (retrieve) =====
def retrieve(state: AgentState):
    """
    현재 질문을 기반으로 관련 문서를 검색합니다.
    """
    print("----- [RETRIEVER] -----")
    question = state["question"]
    # [ 1 ] 현재 질문(또는 재작성된 질문)으로 벡터 DB에서 가장 유사한 문서 3개를 검색합니다.
    relevant_doc = retriever.invoke(question) # [ 1 ]
    context = ""
    # [ 2 ] 나중에 출처를 밝힐 수 있도록 각 문서 조각 앞에 'Page 번호'를 붙여서 하나의 텍스트로 합칩니다.
    # (PDF 페이지 인덱스는 0부터 시작하므로 사람 눈높이에 맞춰 +1을 해줍니다)
    for doc in relevant_doc: # [ 2 ]
        context += f"Page {doc.metadata['page']+1}: {doc.page_content}\n"


    # Tool 호출에 대한 응답 메시지(검색 결과) 생성
    last_message = state["messages"][-1]

    # 만약 직전 메시지가 chatbot의 도구 호출(tool_calls)이었다면 규격에 맞춰 ToolMessage로 포장합니다.
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        tool_call_id = last_message.tool_calls[0]['id']
        tool_message = ToolMessage( # [ 3 ]
            content=context,
            name="retriever",
            tool_call_id=tool_call_id
        )
        return {"messages": [tool_message], "context": context} # [ 4 ]
    else:
        # transform_query(질문 재작성)를 거쳐 다시 온 경우에는 일반 메시지와 context로 업데이트합니다.
        return {"messages": [context], "context": context}


# ===== [부서 3] 자료 정리원 노드 (context_organizer) =====
def context_organizer(state: AgentState):
    """
    검색된 결과를 정리합니다.
    """
    print("----- [CONTEXT ORGANIZER] -----")
    context = state["context"]

    # 비유: PDF에서 긁어온 텍스트는 줄바꿈이 깨져 있거나 공백이 많을 수 있어서,
    # 집필가(generate)가 읽기 편하도록 문서 내용을 삭제하지 않고 깔끔하게 정돈만 해달라고 요청합니다.
    context_organizer_prompt = ChatPromptTemplate.from_messages(
        [
            (   "system",
                """당신은 검색증강생성(RAG)을 위한 검색문서를 정리하는 전문가입니다.
                아래의 검색된 결과 문서를 확인하고, LLM이 해당 문서를 정리된 형태로 참고할 수 있도록
                문서의 불필요한 공백 등을 삭제하거나 정렬을 다시하여 정리된 형태로 반환해주세요.
                내용을 삭제하는 것을 최소로 합니다. 페이지 번호 정보를 절대 삭제하지 마세요."""
            ),
            (
                "user",
                """
                검색 결과: {context}
                """,
            ),
        ]
    )

    # 프롬프트와 AI 모델을 파이프(|)로 연결해 체인(Chain)을 만들고 실행합니다.
    context_organizer = context_organizer_prompt | llm
    organized_context = context_organizer.invoke({"context": context})

    return {"context": organized_context.content, "messages": [AIMessage(organized_context.content)]}



# ===== [부서 4] 질문 교정 전문가 노드 (transform_query) =====
def transform_query(state: AgentState):
    """
    더 나은 질문을 생성하기 위해 쿼리를 변환합니다.

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        state (dict): 재구성된 질문으로 question 키를 업데이트
    """
    # 비유: 도서관에서 "그거 있잖아, 발음 굴러가는 거"라고 물어봐서 원하는 책이 안 나왔을 때,
    # "구개음화의 정의와 표준 발음 규정"처럼 검색이 잘 되는 명확한 질문으로 바꿔주는 부서입니다.
    print("----- [TRANSFORM QUERY] -----")
    question = state["question"]

    system = """
    당신은 질문을 다시 작성하는 전문가입니다. 입력된 질문을 벡터 저장소 검색에 최적화된 더 나은 버전으로 변환하세요.
    입력을 살펴보고 질문의 핵심적인 의미와 의도를 파악하여 개선된 질문을 만들어주세요."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "user",
                "다음은 초기 질문입니다: \n\n {question} \n 한국어로 개선된 질문을 작성해주세요.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm

    better_question = question_rewriter.invoke({"question": question})
    # 재작성된 질문을 저장하고, 무한 반복을 막기 위해 재시도 횟수(retry_num)를 1 증가시킵니다!
    return {"question": better_question.content, "messages": [better_question], "retry_num": state["retry_num"] + 1 if state.get("retry_num") else 1}


# ===== [부서 5] 최종 집필가 노드 (generate) =====
def generate(state: AgentState):
    """
    검색된 문서와 질문을 기반으로 답변을 생성합니다.
    """
    print("----- [GENERATE] -----")
    question = state["question"]
    context = state["context"]

    retry_num = state.get("retry_num", 0)

    # [ 1 ] 만약 3번이나 질문을 바꿔서 재검색했는데도 관련 문서를 못 찾았다면(retry_num >= 3),
    # 억지로 지어내지 않고 정중히 양해를 구한 뒤 현재 문서들로 답할 수 있는 대체 질문들을 추천해 줍니다!
    if retry_num >= 3: # [ 1 ]
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 검색된 문서를 통해 해결할 수 있는 질문을 추출하는 어시스턴트입니다.
                    사용자가 해결하고자 한 질문이 있었으나 검색 컨텍스트가 충분하지 않은 상황이므로, 주어진 검색 결과 내에서 답변할 수 있는 질문을 새롭게 작성해 나열하세요.
                    사용자에게 질문에 대한 답변을 하지 못함에 양해를 구하고, 다른 질문의 기회와 선택지를 제공하는 친절한 가이드를 하세요.
                    """,
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n답변:",
                ),
            ]
        )
    # [ 2 ] 관련 문서를 잘 찾았다면, 검색된 컨텍스트만 사용하여 간결하게 답하고 반드시 출처(페이지 번호)를 적게 합니다.
    else: # [ 2 ]
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 질문-답변 업무를 수행하는 어시스턴트입니다. 검색된 컨텍스트를 사용하여 질문에 답변하세요.
                    답변을 모르는 경우, 모른다고 말하세요.
                    답변은 간결하게 작성하고, 반드시 답변의 출처(페이지 번호)를 함께 명시해주세요.


                    """,
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n답변:",
                ),
            ]
        )

    rag_chain = rag_prompt | llm
    response = rag_chain.invoke({"question": question, "context": context})
    return {"question": question, "answer": response.content, "messages": [response]}
