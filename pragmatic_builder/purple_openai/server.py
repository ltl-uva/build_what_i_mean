import argparse
import logging
import os
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from openai import AsyncOpenAI
import uvicorn

logger = logging.getLogger(__name__)


def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="block_building",
        name="Block building",
        description="Build block on the grid",
        tags=["blocks", "building"],
        examples=[],
    )
    return AgentCard(
        name="openai_purple_agent",
        description="Purple agent powered by OpenAI.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


class OpenAIPurpleAgent(AgentExecutor):
    def __init__(self, debug: bool = False):
        self._debug = debug
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        if self._debug:
            logger.info("-----")
            logger.info("User input: %s", user_input)

        if not self._api_key:
            response = "[ASK];missing OPENAI_API_KEY"
            await event_queue.enqueue_event(
                new_agent_text_message(response, context_id=context.context_id)
            )
            return

        system_prompt = (
            "You are a purple agent that builds blocks on a grid. "
            "Respond with one line only. Use either "
            "[BUILD];Color,x,y,z;Color,x,y,z;... or "
            "[ASK];<question> if you need clarification."
        )
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            content = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("OpenAI request failed: %s", exc)
            content = "[ASK];error from model"

        await event_queue.enqueue_event(
            new_agent_text_message(content, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9022, help="Port to bind the server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    debug_env = os.getenv("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug = args.debug or debug_env
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING)

    card = prepare_agent_card(f"http://{args.host}:{args.port}/")
    request_handler = DefaultRequestHandler(
        agent_executor=OpenAIPurpleAgent(debug=debug),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
