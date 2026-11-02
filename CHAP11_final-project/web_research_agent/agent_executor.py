# ==============================================================================
# 📘 [CHAP11] 웹 리서치 에이전트 A2A 실행자 (WebResearchAgentExecutor)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - WebResearchAgent(외부 특파원)가 A2A 프로토콜로 취재 요청을 받아 검색을 수행하고,
#   완성된 기사를 `web_search_result` 아티팩트 상자에 담아 오케스트레이터에게 전달하는 매니저입니다.
# ==============================================================================

import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message, new_task
from agent import WebResearchAgent

logger = logging.getLogger(__name__)


class WebResearchAgentExecutor(AgentExecutor):
    """
    Web Research 에이전트 A2A 실행자
    """

    def __init__(self):
        self.agent = WebResearchAgent()
        self.initialized = False

    async def execute( # [ 1 ]
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """에이전트 실행 및 스트리밍 결과 전송"""
        if not self.initialized:
            try:
                await self.agent.initialize()
                self.initialized = True
                logger.info("[WEB AGENT] [EXECUTOR] 초기화 완료")
            except Exception as e:
                logger.error(f"[WEB AGENT] [ERROR] 초기화 실패: {e}")
                raise

        query = context.get_user_input()
        logger.info(f"[WEB AGENT] [EXECUTOR] 쿼리 수신: {query[:100]}...")

        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for item in self.agent.stream(query): # [ 2 ]
                is_complete = item.get('is_task_complete', False)
                require_input = item.get('require_user_input', False)
                content = item.get('content', '')

                if not is_complete and not require_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                elif is_complete: # [ 3 ]
                    await updater.add_artifact(
                        parts=[Part(root=TextPart(text=content))],
                        name='web_search_result'
                    )
                    await updater.complete()
                    break

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"오류 발생: {str(e)}",
                    task.context_id,
                    task.id,
                ),
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """작업 취소"""
        task = context.current_task
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel(
            new_agent_text_message(
                "웹 검색 작업이 취소되었습니다.",
                task.context_id,
                task.id,
            )
        )
