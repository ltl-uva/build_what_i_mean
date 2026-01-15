import asyncio
import logging
import sys
from pathlib import Path
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidParamsError,
    TaskState,
    Part,
    TextPart,
)
from a2a.utils import (
    new_agent_text_message,
)
from a2a.utils.errors import ServerError

from building_task import BuildingGameTask
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from agentbeats.conversation_recorder import ConversationRecorder
from agentbeats.question_answerer import QuestionAnswerer

logger = logging.getLogger(__name__)


class BuildingInstructorGreenAgent:
    def __init__(self, debug: bool = False, transcript_path: str | None = None):
        self._required_roles = ["rita"]
        self._required_config_keys = ["list1_path", "list2_path"]
        self._tool_provider = ToolProvider()
        self._debug = debug
        self._recorder = ConversationRecorder(transcript_path) if transcript_path else None
        self._qa = QuestionAnswerer.from_env()

    async def _debug_pause(self, prompt: str) -> None:
        if not self._debug or not sys.stdin.isatty():
            return
        await asyncio.to_thread(input, prompt)

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting structure evaluation: {req}")

        list1_path = self._resolve_path(req.config["list1_path"])
        list2_path = self._resolve_path(req.config["list2_path"])
        bulding_task = BuildingGameTask(list1_path, list2_path)

        trials = bulding_task.run(None)
        logger.info(f"created trials {trials}")

        async def turn(role: str, prompt: str) -> str:
            if self._recorder:
                self._recorder.record(f"GREEN -> {role}: {prompt}")
            response = await self._tool_provider.talk_to_agent(prompt, str(req.participants[role]),
                                                               new_conversation=False)
            logger.info(f"{role}: {response}")
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            if self._recorder:
                self._recorder.record(f"{role} -> GREEN: {response}")
            await self._debug_pause("Press Enter to continue...\n")
            return response

        async def send_feedback(role: str, message: str) -> None:
            if self._recorder:
                self._recorder.record(f"GREEN -> {role}: {message}")
            await self._tool_provider.talk_to_agent(
                message,
                str(req.participants[role]),
                new_conversation=False,
            )
            await updater.update_status(TaskState.working, new_agent_text_message(message))
            await self._debug_pause("Press Enter to continue...\n")

        results = {}
        num_correct = 0
        scored_count = 0
        questions_count = 0
        # TODO: initial turn with the grid context
        task_description = f"[TASK_DESCRIPTION] {trials['grid_context']})"
        purple_agent_role = 'rita'
        for speaker in [trials["instructions_A"], trials["instructions_B"]]:
            prompt_chain = []
            response_chain = []
            for instruction in speaker:
                round_questions_count = 0
                prompt = f"{task_description}\n[START_STRUCTURE] {instruction['start_structure']}\n{instruction['instruction']}"
                prompt_chain.append(prompt)
                built = False
                eval_result = {}
                while built is not True:
                    instruction_response = await turn(purple_agent_role, prompt)
                    response_chain.append(instruction_response)
                    eval_result = await self.eval_message(
                        instruction_response,
                        instruction["target_structure"],
                    )
                    round_questions_count += eval_result["num_questions"]
                    prompt = eval_result['message']
                    built = eval_result['built']
                await send_feedback(purple_agent_role, f"Feedback: {eval_result['message']}")
                if eval_result["num_correct"] is not None:
                    scored_count += 1
                    num_correct += eval_result["num_correct"]
                questions_count += round_questions_count
                results[instruction["round"]] = {"prompts": prompt_chain,
                                                   "responses": response_chain,
                                                   "eval_feedback_message": eval_result["message"],
                                                   "num_correct": eval_result["num_correct"],
                                                    "num_questions": round_questions_count,
                                                   "response_feedback": None}


        accuracy = (num_correct / scored_count * 100.0) if scored_count else 0.0
        questions_count = questions_count/len(trials["instructions_A"] + trials["instructions_B"])
        # TODO: metric here to compare response to expected answer
        try:
            result = EvalResult(accuracy=accuracy, avg_questions_per_instruction=questions_count)
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=result.model_dump_json())),
                ],
                name="Result",
            )
        finally:
            self._tool_provider.reset()
        return result

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        return True, "ok"

    async def eval_message(self, response: str, target_structure: str):
        string_response = response.split(";")
        action = string_response[0]
        match action:
            case "[BUILD]":
                content = self._normalize_structure(string_response[1:])
                target_structure_set = self._normalize_structure(target_structure.split(";"))
                if content == target_structure_set:
                    return {"message": f"Correct structure built. {target_structure}",
                            "num_correct": 1,
                            "num_questions":0,
                            "built": True
                            }
                else:
                    return {"message":"Incorrect structure. Expected: {target_structure}, but got: {';'.join(content)}",
                            "num_correct": 0,
                            "num_questions":0,
                            "built":True
                            }
            case "[ASK]":
                content = ";".join(string_response[1:]).strip()
                if self._qa:
                    answer = await self._qa.answer(
                        question=content,
                        target_structure=target_structure,
                    )
                else:
                    answer = self._fallback_answer(content, target_structure)
                return {"message": f"Answer: {answer}",
                        "num_correct":None,
                        "num_questions": 1,
                        "built": False}
            case _:
                raise ServerError(error=InvalidParamsError(message="Invalid action in response"))

    @staticmethod
    def _fallback_answer(question: str, target_structure: str) -> str:
        colors = []
        for block in target_structure.split(";"):
            if block:
                colors.append(block.split(",", 1)[0])
        unique_colors = sorted(set(colors))
        if "color" in question.lower() and unique_colors:
            return f"Colors in target: {', '.join(unique_colors)}."
        return "I can answer questions about the target structure."

    @staticmethod
    def _normalize_structure(items) -> set[str]:
        # Normalize color casing and strip empty/invalid entries for stable comparisons.
        normalized = set()
        for item in items:
            item = item.strip()
            if not item:
                continue
            parts = item.split(",")
            if len(parts) != 4:
                continue
            color = parts[0].strip().capitalize()
            coords = [p.strip() for p in parts[1:]]
            normalized.add(",".join([color, *coords]))
        return normalized

    @staticmethod
    def _resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute() or path.exists():
            return str(path)
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / path
        return str(candidate)
