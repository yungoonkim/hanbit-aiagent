# ==============================================================================
# 📘 [CHAP11] 오케스트레이터 에이전트 A2A 실행자 (OrchestratorAgentExecutor)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - 오케스트레이터(총괄 PM) 역시 외부 클라이언트(`test_client.py` 등)에서 볼 때는
#   하나의 A2A 에이전트 서버(10010번 포트)입니다!
# - 즉, 고객에게는 A2A 서버로서 주문을 받고, 산하 3개 에이전트에게는 A2A 클라이언트로서
#   업무를 지시하는 '양방향 A2A 허브' 역할을 수행합니다.
# ==============================================================================

import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message, new_task
from agent import OrchestratorAgent

logger = logging.getLogger(__name__)


class OrchestratorAgentExecutor(AgentExecutor):
    """
    Orchestrator 에이전트 A2A 실행자
    """

    def __init__(self):
        self.agent = OrchestratorAgent()
        self.initialized = False

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """오케스트레이터 실행 및 단계별 상태 중계"""
        # 초기화
        if not self.initialized:
            try:
                await self.agent.initialize()
                self.initialized = True
                logger.info("[ORCHESTRATOR] [EXECUTOR] 초기화 완료")
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] [ERROR] 초기화 실패: {e}")
                raise

        # 사용자 입력 가져오기
        query = context.get_user_input()
        logger.info(f"[ORCHESTRATOR] [EXECUTOR] 쿼리 수신: {query[:100]}...")

        # 태스크 생성
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        session_id = task.context_id

        try: # [ 1 ]
            async for item in self.agent.stream(query, session_id):
                is_complete = item.get('is_task_complete', False)
                require_input = item.get('require_user_input', False)
                content = item.get('content', '')
                artifacts = item.get('artifacts', [])

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
                elif is_complete: # [ 2 ]
                    # 최종 통합 답변을 orchestrator_result 아티팩트로 추가하고 완료 처리
                    await updater.add_artifact(
                        parts=[Part(root=TextPart(text=content))],
                        name='orchestrator_result'
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
                "작업이 취소되었습니다.",
                task.context_id,
                task.id,
            )
        )
