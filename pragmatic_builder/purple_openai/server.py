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
            "You are a block-building agent on a 9x9 grid.\n\n"
            
            "GRID COORDINATES:\n"
            "- The grid is the x-z plane. Origin (0,0) is the center.\n"
            "- Valid x,z coordinates: [-400,-300,-200,-100,0,100,200,300,400]\n"
            "- Y-axis is vertical (height). Ground level y=50. Each block adds +100.\n"
            "- Valid y coordinates: [50,150,250,350,450]\n"
            "- Format: Color,x,y,z (e.g., Red,0,50,0 means a red block at center, ground level)\n\n"
            
            "YOUR RESPONSES:\n"
            "You must respond with ONLY one of these two formats:\n\n"
            
            "1. [BUILD];Color,x,y,z;Color,x,y,z;...\n"
            "   - Use this to build or modify the structure\n"
            "   - List ALL blocks that should be on the grid (including existing ones from START_STRUCTURE)\n"
            "   - No spaces, semicolons separate blocks\n"
            "   - Colors must be capitalized (Red, Blue, Green, Yellow, Purple, etc.)\n"
            "   Example: [BUILD];Purple,0,50,0;Green,0,150,0;Green,0,250,0\n\n"
            
            "2. [ASK];<your question>\n"
            "   - Use this if you need clarification about the instruction\n"
            "   - Costs -5 points, so only ask if truly necessary\n"
            "   Example: [ASK];How many green blocks should I add?\n\n"
            
            "IMPORTANT:\n"
            "- Never respond with 'Acknowledged' or any other text\n"
            "- Always respond with either [BUILD] or [ASK]\n"
            "- When building, include the START_STRUCTURE blocks plus any new blocks from the instruction\n"
            "- Correct structure: +10 points. Incorrect: -10 points. Question: -5 points.\n"
            "- Learn from feedback in previous messages to improve your performance."
        )
        try:
            # Prepare API call parameters
            api_params = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                "temperature": 0.2,
            }
            
            # GPT-4o and newer models use max_completion_tokens instead of max_tokens
            if "gpt-4o" in self._model or "gpt-4-turbo" or "gpt-5.1" in self._model:
                api_params["max_completion_tokens"] = 512
            else:
                api_params["max_tokens"] = 512
            
            completion = await self._client.chat.completions.create(**api_params)
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
    parser.add_argument("--card-url", default="", help="URL for the agent card")
    args = parser.parse_args()

    debug_env = os.getenv("AGENT_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    debug = args.debug or debug_env
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING)

    card_url = args.card_url
    if not card_url:
        if args.host == "0.0.0.0":
            card_host = "127.0.0.1"
        else:
            card_host = args.host
        card_url = f"http://{card_host}:{args.port}"

    card = prepare_agent_card(card_url)
    request_handler = DefaultRequestHandler(
        agent_executor=OpenAIPurpleAgent(debug=debug),
        task_store=InMemoryTaskStore(),
    )

    logger.info(f"Starting OpenAI purple agent on {args.host}:{args.port} with card URL: {card_url}")
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
