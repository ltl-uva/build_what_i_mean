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
        name="openrouter_purple_agent",
        description="Purple agent powered by OpenRouter.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


class OpenRouterPurpleAgent(AgentExecutor):
    def __init__(self, debug: bool = False):
        self._debug = debug
        # Default to Claude Sonnet 4.5, but can be overridden
        self._model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
        self._api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        # OpenRouter base URL
        self._base_url = "https://openrouter.ai/api/v1"
        
        # Optional: Site URL and App Name for OpenRouter rankings
        self._site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        self._app_name = os.getenv("OPENROUTER_APP_NAME", "Purple Agent").strip()
        
        # Set up headers for OpenRouter
        extra_headers = {}
        if self._site_url:
            extra_headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            extra_headers["X-Title"] = self._app_name
        
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            default_headers=extra_headers if extra_headers else None
        )
        
        # Conversation history - reset between evaluations
        self._conversation_history = []
        self._max_history_length = 40  # Keep last 40 messages to avoid token limits
        
        # Error tracking - fail fast if too many errors
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        if self._debug:
            logger.info("-----")
            logger.info("User input: %s", user_input)
            logger.info("Using model: %s", self._model)
            logger.info("Conversation history length: %d", len(self._conversation_history))

        if not self._api_key:
            response = "[ASK];missing OPENROUTER_API_KEY"
            await event_queue.enqueue_event(
                new_agent_text_message(response, context_id=context.context_id)
            )
            return

        # Check if this is a new task/evaluation starting - reset history and send system prompt
        is_new_task = "A new task is starting" in user_input or len(self._conversation_history) == 0
        
        if "A new task is starting" in user_input:
            if self._debug:
                logger.info("New task detected - resetting conversation history")
            self._conversation_history = []
            self._consecutive_errors = 0  # Reset error counter on new task
            is_new_task = True

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
        
        # Build messages array
        messages = []
        
        # Only add system prompt if this is the first message (empty history)
        if is_new_task:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history (keep last N messages to avoid token limits)
        if len(self._conversation_history) > self._max_history_length:
            # Keep only recent history
            self._conversation_history = self._conversation_history[-self._max_history_length:]
        
        messages.extend(self._conversation_history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )
            content = (completion.choices[0].message.content or "").strip()
            self._consecutive_errors = 0  # Reset error counter on success
            
            if self._debug:
                logger.info("Model response: %s", content)
                
        except Exception as exc:
            self._consecutive_errors += 1
            logger.error(f"OpenRouter request failed ({self._consecutive_errors}/{self._max_consecutive_errors}): {exc}")
            
            # If too many consecutive errors, raise exception to stop the evaluation
            if self._consecutive_errors >= self._max_consecutive_errors:
                error_msg = f"Too many consecutive API errors ({self._consecutive_errors}). Check: API key, model name ({self._model}), network connection."
                logger.error(error_msg)
                content = f"[ASK];CRITICAL ERROR - {error_msg}"
            else:
                content = "[ASK];API error - retrying"

        # Add this exchange to conversation history
        self._conversation_history.append({"role": "user", "content": user_input})
        self._conversation_history.append({"role": "assistant", "content": content})

        await event_queue.enqueue_event(
            new_agent_text_message(content, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenRouter purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9023, help="Port to bind the server")
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
        agent_executor=OpenRouterPurpleAgent(debug=debug),
        task_store=InMemoryTaskStore(),
    )

    logger.info(f"Starting OpenRouter purple agent on {args.host}:{args.port} with card URL: {card_url}")
    logger.info(f"Using model: {os.getenv('OPENROUTER_MODEL', 'anthropic/claude-sonnet-4.5')}")
    logger.info("=" * 60)
    logger.info("Agent is starting up...")
    logger.info("Once you see 'Application startup complete', the agent is ready!")
    logger.info("=" * 60)
    
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=600,  # Increased from 300 to 600 seconds (10 minutes)
    )


if __name__ == "__main__":
    main()
