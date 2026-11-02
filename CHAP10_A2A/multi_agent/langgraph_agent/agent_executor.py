# ==============================================================================
# 📘 [CHAP10] LangGraph 에이전트 A2A 실행자 (LangGraphAgentExecutor)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - TaskUpdater = 택배 배송 상태 알림판!
#   작업 중일 때는 `TaskState.working`(배송 준비 중),
#   추가 정보가 필요할 때는 `TaskState.input_required`(수취인 확인 필요),
#   작업이 끝나면 결과물을 상자(`add_artifact`)에 담아 `complete()`(배송 완료) 도장을 찍습니다!
# ==============================================================================

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message, new_task
from agent import LangGraphAgent


class LangGraphAgentExecutor(AgentExecutor): # [ 1 ]
    """
    LangGraphAgent를 A2A 프로토콜 서버와 연결해주는 실행자(어댑터) 클래스
    """
    def __init__(self):
        self.agent = LangGraphAgent()

    async def execute( # [ 2 ]
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        A2A 클라이언트(또는 오케스트레이터)의 요청을 받아 LangGraphAgent를 실행합니다.
        """
        # 1. 요청 컨텍스트에서 사용자가 보낸 질문 텍스트를 꺼냅니다.
        query = context.get_user_input()

        # 2. 현재 진행 중인 태스크(Task)가 없으면 새 작업 번호표(new_task)를 발급합니다.
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        # 3. 작업 상태와 결과물(Artifact)을 실시간으로 보고할 TaskUpdater를 준비합니다.
        updater = TaskUpdater(event_queue, task.id, task.context_id) # [ 3 ]

        session_id = task.context_id

        try:
            # 4. LangGraphAgent를 스트리밍으로 실행하며 단계별 이벤트를 처리합니다.
            async for item in self.agent.stream(query, session_id): # [ 4 ]
                is_task_complete = item.get('is_task_complete', False)
                require_user_input = item.get('require_user_input', False)
                content = item.get('content', '')

                # [상태 1] 아직 작업 진행 중일 때 -> 'working(작업 중)' 상태 전송
                if not is_task_complete and not require_user_input: # [ 5 ]
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                    )
                # [상태 2] 사용자의 추가 답변이 필요할 때 -> 'input_required' 상태 전송 후 대기
                elif require_user_input: # [ 6 ]
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
                # [상태 3] 작업이 최종 완료되었을 때 -> 결과물(Artifact)을 포장해 넣고 완료 처리!
                elif is_task_complete: # [ 7 ]
                    await updater.add_artifact(
                        parts=[Part(root=TextPart(text=content))],
                        name='agent_result'
                    )

                    await updater.complete()
                    break

        except Exception as e: # [ 8 ]
            # 예외 발생 시 'failed(실패)' 상태로 보고합니다.
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"에러 발생: {str(e)}",
                    task.context_id,
                    task.id,
                ),
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """클라이언트가 작업 취소를 요청했을 때 상태를 canceled로 변경합니다."""
        task = context.current_task

        # TaskUpdater를 사용하여 취소 상태로 업데이트
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel(
            new_agent_text_message(
                "작업이 취소되었습니다.",
                task.context_id,
                task.id,
            )
        )
