# ==============================================================================
# 📘 [CHAP11] 사내 지식 RAG 에이전트 A2A 실행자 (InternalRAGAgentExecutor)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - InternalRAGAgent(사내 도서관 수석 사서)가 A2A 프로토콜로 검색/인덱싱 요청을 받아
#   노드별 진행 상황("🎯 요청 분석 중...", "📊 벡터 유사도 검색 중...")을 실시간으로 중계하고,
#   최종 답변을 `rag_answer` 아티팩트 상자에 담아 돌려주는 실행 매니저입니다.
# ==============================================================================

import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message, new_task
from agent import InternalRAGAgent

logger = logging.getLogger(__name__)


class InternalRAGAgentExecutor(AgentExecutor): # [ 1 ]
    """Internal RAG 에이전트 A2A 실행자"""

    def __init__(self):
        self.agent = InternalRAGAgent()
        self.initialized = False

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """에이전트 실행 및 스트리밍 결과 전송"""
        if not self.initialized:
            try:
                self.initialized = True
                logger.info("[RAG AGENT] [EXECUTOR] 초기화 완료")
            except Exception as e:
                logger.error(f"[RAG AGENT] [ERROR] 초기화 실패: {e}")
                raise

        query = context.get_user_input()
        logger.info(f"[RAG AGENT] [EXECUTOR] 쿼리 수신: {query[:100]}...")

        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        async for item in self.agent.stream(query): # [ 2 ]
            is_complete = item.get('is_task_complete', False)
            require_input = item.get('require_user_input', False)
            content = item.get('content', '')

            if not is_complete and not require_input:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(content, task.context_id, task.id),
                )
            elif require_input:
                await updater.update_status(
                    TaskState.input_required,
                    new_agent_text_message(content, task.context_id, task.id),
                    final=True,
                )
                break
            elif is_complete:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=content))],
                    name='rag_answer'
                )
                await updater.complete()
                break

    async def cancel( # [ 3 ]
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """작업 취소"""
        task = context.current_task
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel(
            new_agent_text_message(
                "RAG 작업이 취소되었습니다.",
                task.context_id,
                task.id,
            )
        )
