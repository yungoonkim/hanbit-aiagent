# ==============================================================================
# 📘 [CHAP11] 파일 관리 에이전트 A2A 실행자 (FileManagementAgentExecutor)
# ==============================================================================
# 💡 일상 비유로 이해하기:
# - 파일 관리 에이전트가 작업을 마치면 두 종류의 상자(Artifact)를 포장합니다:
#   1) 사람이나 LLM이 읽기 좋은 설명글 상자 (`TextPart`: 'file_operation_result')
#   2) 오케스트레이터와 RAG 에이전트가 프로그램적으로 파일 ID(`storage_ref`)를 꺼내 쓸 수 있는
#      구조화된 목록 상자 (`DataPart`: 'file_list')
# ==============================================================================

import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart, DataPart
from a2a.utils import new_agent_text_message, new_task
from agent import FileManagementAgent

logger = logging.getLogger(__name__)


class FileManagementAgentExecutor(AgentExecutor):
    """
    File Management 에이전트 A2A 실행자
    """

    def __init__(self):
        self.agent = FileManagementAgent()
        self.initialized = False

    async def execute( # [ 1 ]
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # 최초 호출 시 Google Drive 인증 및 에이전트 초기화
        if not self.initialized:
            try:
                await self.agent.initialize()
                self.initialized = True
                logger.info("[FILE AGENT] [EXECUTOR] 초기화 완료")
            except Exception as e:
                logger.error(f"[FILE AGENT] [ERROR] 초기화 실패: {e}")
                raise

        # 사용자(또는 오케스트레이터)가 보낸 요청 추출
        query = context.get_user_input() # [ 2 ]
        logger.info(f"[FILE AGENT] [EXECUTOR] 쿼리 수신: {query[:100]}...")

        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for item in self.agent.stream(query): # [ 3 ]
                is_complete = item.get('is_task_complete', False)
                require_input = item.get('require_user_input', False)
                content = item.get('content', '')

                # 작업 진행 중 상태 업데이트
                if not is_complete and not require_input: # [ 4 ]
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
                elif is_complete:
                    data = item.get('data') # [ 5 ]

                    # 파일 목록 조회가 포함된 경우 DataPart 아티팩트로 구조화 데이터 추가!
                    # (오케스트레이터가 이를 받아 RAG 에이전트에게 파일 storage_ref를 전달합니다)
                    if data and 'files' in data:
                        await updater.add_artifact(
                            parts=[Part(root=DataPart(data=data))],
                            name='file_list'
                        )
                        logger.info(f"[FILE AGENT] [EXECUTOR] 파일 목록 DataPart 추가: {len(data.get('files', []))}개")

                    # 텍스트 결과 아티팩트 추가
                    await updater.add_artifact(
                        parts=[Part(root=TextPart(text=content))],
                        name='file_operation_result'
                    )

                    await updater.complete()
                    break

        except Exception as e:
            await updater.update_status( # [ 6 ]
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
                "파일 작업이 취소되었습니다.",
                task.context_id,
                task.id,
            )
        )
