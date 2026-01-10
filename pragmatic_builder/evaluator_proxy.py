import argparse
import contextlib
import uvicorn
import asyncio
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidParamsError,
    Task,
    UnsupportedOperationError,
    InternalError,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskState,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from pydantic import ValidationError
from green_agent import BuildingInstructorGreenAgent
from agentbeats.models import EvalRequest

logger = logging.getLogger(__name__)

class GreenExecutor(AgentExecutor):
    def __init__(self, green_agent: BuildingInstructorGreenAgent):
        self.agent = green_agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        request_text = context.get_user_input()

        try:
            req: EvalRequest = EvalRequest.model_validate_json(request_text)
            ok, msg = self.agent.validate_request(req)
            if not ok:
                raise ServerError(error=InvalidParamsError(message=msg))
        except ValidationError as e:
            print(e)
            raise ServerError(error=InvalidParamsError(message=e.json()))
        #
        msg = context.message
        if msg:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        else:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting assessment.\n{req.model_dump_json()}", context_id=context.context_id)
        )

        try:
            await self.agent.run_eval(req, updater)
            await updater.complete()
        except Exception as e:
            print(f"Agent error: {e}")
            await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context.context_id))
            raise ServerError(error=InternalError(message=str(e)))

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


def instruction_following_evaluator_card(agent_name: str, card_url: str) -> AgentCard:
    skill = AgentSkill(
        id='evaluation_instruction_following',
        name='Evaluate built structure',
        description='Gives instruction and evaluate how the builder follow instructions',
        tags=['instructor','building'],
        examples=["""
            {
              "participants": {
                "builder": "https://builder.example.com:443"
              },
              "config":{
              "domain":"blocks"}"""
            ]

    )
    agent_card = AgentCard(
        name=agent_name,
        description='Gives instruction and evaluate how the builder follow instructions',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return agent_card

async def main():
    parser = argparse.ArgumentParser(description="Run the builder evaluator agent")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    args = parser.parse_args()

    agent_url_cm = contextlib.nullcontext(f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = BuildingInstructorGreenAgent()
        executor = GreenExecutor(agent)
        agent_card = instruction_following_evaluator_card("StructureEvaluator", agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()

if __name__ == '__main__':
    asyncio.run(main())