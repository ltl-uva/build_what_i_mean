import asyncio
import json
import logging
import os
import sys
from datetime import datetime
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
        
        # NEW: Setup checkpoint directory
        self._checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "logs/checkpoints"))
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # NEW: Track model version
        self._model_version = None

    async def _debug_pause(self, prompt: str) -> None:
        if not self._debug or not sys.stdin.isatty():
            return
            return
        await asyncio.to_thread(input, prompt)

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting structure evaluation: {req}")
        
        # NEW: Get model version from environment
        self._model_version = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "unknown"
        model_name_clean = self._model_version.replace("/", "_").replace(":", "_")
        logger.info(f"Using model: {self._model_version}")
        
        # NEW: Create run-specific checkpoint file with model name
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self._checkpoint_dir / f"{model_name_clean}_eval_{run_id}.json"

        list1_path = self._resolve_path(req.config["list1_path"])
        list2_path = self._resolve_path(req.config["list2_path"])

        # Run 8 trials with different seeds
        num_seeds = 2
        all_accuracies = []
        all_avg_questions = []
        all_results = {}

        purple_agent_role = 'rita'
        
        # NEW: Load existing checkpoint if exists
        start_seed = 0
        if checkpoint_file.exists():
            try:
                checkpoint_data = json.loads(checkpoint_file.read_text())
                all_results = checkpoint_data.get("results", {})
                start_seed = len(all_results)
                logger.info(f"Resuming from checkpoint at seed {start_seed}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}. Starting from seed 0.")

        async def turn(role: str, prompt: str) -> str:
            if self._recorder:
                self._recorder.record(f"GREEN -> {role}: {prompt}")
            
            try:
                response = await self._tool_provider.talk_to_agent(
                    prompt, 
                    str(req.participants[role]),
                    new_conversation=False
                )
            except Exception as e:
                logger.error(f"Error communicating with {role}: {e}")
                # Save checkpoint before re-raising
                self._save_checkpoint(checkpoint_file, all_results, run_id)
                raise
            
            logger.info(f"{role}: {response}")
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            if self._recorder:
                self._recorder.record(f"{role} -> GREEN: {response}")
            # Removed debug pause to prevent hangs
            return response

        async def send_feedback(role: str, message: str) -> None:
            if self._recorder:
                self._recorder.record(f"GREEN -> {role}: {message}")
            # Send feedback and capture response (but ignore it - we don't need acknowledgments)
            try:
                response = await self._tool_provider.talk_to_agent(
                    message,
                    str(req.participants[role]),
                    new_conversation=False,
                )
                # Log the response for debugging but don't process it
                logger.info(f"Feedback acknowledgment from {role}: {response}")
                if self._recorder:
                    self._recorder.record(f"{role} -> GREEN (feedback ack): {response}")
            except Exception as e:
                logger.warning(f"Error sending feedback to {role}: {e}")
            
            await updater.update_status(TaskState.working, new_agent_text_message(message))
            # Removed debug pause to prevent hangs

        # Main evaluation loop with checkpointing
        for seed in range(start_seed, num_seeds):
            logger.info(f"Starting trial {seed + 1}/{num_seeds} with seed {seed}")

            try:

                pending_header = []

                if seed > 0:
                    pending_header.append("A new task is starting, now you will play the game again.")

                building_task = BuildingGameTask(list1_path, list2_path, seed=seed)
                trials = building_task.run(None)
                logger.info(f"Created trials for seed {seed}: {trials}")

                results = {}
                num_correct = 0
                scored_count = 0
                questions_count = 0
                total_score = 0  # Track cumulative score
                
                # Track statistics by speaker and trial type
                stats_by_speaker = {"Lisa": {"questions": 0, "trials": 0}, "Pia": {"questions": 0, "trials": 0}}
                stats_by_trial_type = {
                    "fully_spec": {"questions": 0, "trials": 0},
                    "underspec": {"questions": 0, "trials": 0}
                }

                # Prepend task description once per seed
                pending_header.append(f"[TASK_DESCRIPTION] {trials['grid_context']}")

                for speaker_idx, speaker in enumerate([trials["instructions_A"], trials["instructions_B"]]):
                    # MODIFIED: Send speaker transition message (except for the first speaker)
                    if speaker_idx > 0:
                        pending_header.append("Now you will play with a different speaker.")
                    
                    prompt_chain = []
                    response_chain = []
                    for instr_idx, instruction in enumerate(speaker):
                        round_questions_count = 0
                        round_score = 0  # Track score for this round
                        speaker_name = instruction['speaker']
                        # MODIFIED: Don't include task description - only speaker and instruction
                        base_prompt = f"[SPEAKER] {speaker_name}\n[START_STRUCTURE] {instruction['start_structure']}\n{instruction['instruction']}"
                        
                        # Prepend buffered messages to create full context
                        if pending_header:
                            full_context = "\n".join(pending_header) + "\n\n"
                            pending_header = []
                        else:
                            full_context = ""
                        
                        # Build initial prompt with context
                        prompt = full_context + base_prompt
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
                            round_score += eval_result.get("points", 0)  # Accumulate points
                            
                            # For retries, use base prompt + feedback (agent already got context on first attempt)
                            prompt = base_prompt + "\n\n" + eval_result['message']
                            built = eval_result['built']


                        total_score += round_score  # Add round score to total

                        # Create feedback message
                        feedback_msg = f"Feedback: {eval_result['message']} | Round score: {round_score:+d} | Total score: {total_score:+d}"
                        logger.info(f"Round {instruction['round']} feedback: {feedback_msg}")
                        if self._recorder:
                            self._recorder.record(f"GREEN (internal): {feedback_msg}")

                        # Determine if this is the last instruction across all speakers
                        is_last_speaker = speaker_idx == 1  # Second speaker (index 1)
                        is_last_instruction_in_speaker = instr_idx == len(speaker) - 1
                        is_last_instruction_overall = is_last_speaker and is_last_instruction_in_speaker
                        
                        if is_last_instruction_overall:
                            # Last instruction: send feedback as standalone message
                            await turn(purple_agent_role, feedback_msg)
                        else:
                            # Not last instruction: prepend feedback to next instruction
                            pending_header.append(feedback_msg)

                        # Track statistics by speaker
                        stats_by_speaker[speaker_name]["questions"] += round_questions_count
                        stats_by_speaker[speaker_name]["trials"] += 1
                        
                        # Track statistics by trial type
                        trial_id = instruction["trial_id"]
                        # Check if trial_id ends with 'a' or 'b' (underspecified) or is just a number (fully specified)
                        if trial_id[-1] in ['a', 'b']:
                            trial_type = "underspec"
                        else:
                            trial_type = "fully_spec"
                        stats_by_trial_type[trial_type]["questions"] += round_questions_count
                        stats_by_trial_type[trial_type]["trials"] += 1

                        if eval_result["num_correct"] is not None:
                            scored_count += 1
                            num_correct += eval_result["num_correct"]
                        questions_count += round_questions_count
                        results[instruction["round"]] = {
                            "prompts": prompt_chain,
                            "responses": response_chain,
                            "eval_feedback_message": eval_result["message"],
                            "num_correct": eval_result["num_correct"],
                            "num_questions": round_questions_count,
                            "response_feedback": None,
                            "speaker": speaker_name,
                            "round_score": round_score  # Store round score
                        }

                # Calculate metrics for this seed
                accuracy = (num_correct / scored_count * 100.0) if scored_count else 0.0
                avg_questions = questions_count / len(trials["instructions_A"] + trials["instructions_B"])

                all_accuracies.append(accuracy)
                all_avg_questions.append(avg_questions)
                all_results[f"seed_{seed}"] = {
                    "accuracy": accuracy,
                    "avg_questions_per_instruction": avg_questions,
                    "total_score": total_score,
                    "stats_by_speaker": stats_by_speaker,
                    "stats_by_trial_type": stats_by_trial_type,
                    "results": results
                }

                logger.info(
                    f"Seed {seed} - Accuracy: {accuracy:.2f}%, Avg Questions: {avg_questions:.2f}, Total Score: {total_score}")
                logger.info(f"  Lisa: {stats_by_speaker['Lisa']['questions']} questions in {stats_by_speaker['Lisa']['trials']} trials")
                logger.info(f"  Pia: {stats_by_speaker['Pia']['questions']} questions in {stats_by_speaker['Pia']['trials']} trials")
                logger.info(f"  Fully specified: {stats_by_trial_type['fully_spec']['questions']} questions in {stats_by_trial_type['fully_spec']['trials']} trials")
                logger.info(f"  Underspecified: {stats_by_trial_type['underspec']['questions']} questions in {stats_by_trial_type['underspec']['trials']} trials")

                # NEW: Save checkpoint after each seed
                self._save_checkpoint(checkpoint_file, all_results, run_id)
                
            except Exception as e:
                logger.error(f"Error in seed {seed}: {e}")
                # Save checkpoint before re-raising
                self._save_checkpoint(checkpoint_file, all_results, run_id)
                raise

        # Calculate overall averages
        overall_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        overall_avg_questions = sum(all_avg_questions) / len(all_avg_questions) if all_avg_questions else 0.0

        # Calculate average score across all seeds
        all_scores = [all_results[f"seed_{seed}"]["total_score"] for seed in range(start_seed, num_seeds)]
        overall_avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Aggregate statistics by speaker and trial type across all seeds
        aggregate_speaker_stats = {"Lisa": {"questions": 0, "trials": 0}, "Pia": {"questions": 0, "trials": 0}}
        aggregate_trial_type_stats = {"fully_spec": {"questions": 0, "trials": 0}, "underspec": {"questions": 0, "trials": 0}}
        
        for seed in range(start_seed, num_seeds):
            seed_key = f"seed_{seed}"
            if seed_key in all_results:
                for speaker in ["Lisa", "Pia"]:
                    aggregate_speaker_stats[speaker]["questions"] += all_results[seed_key]["stats_by_speaker"][speaker]["questions"]
                    aggregate_speaker_stats[speaker]["trials"] += all_results[seed_key]["stats_by_speaker"][speaker]["trials"]
                
                for trial_type in ["fully_spec", "underspec"]:
                    aggregate_trial_type_stats[trial_type]["questions"] += all_results[seed_key]["stats_by_trial_type"][trial_type]["questions"]
                    aggregate_trial_type_stats[trial_type]["trials"] += all_results[seed_key]["stats_by_trial_type"][trial_type]["trials"]

        logger.info(
            f"Overall - Accuracy: {overall_accuracy:.2f}%, Avg Questions: {overall_avg_questions:.2f}, Avg Score: {overall_avg_score:.2f}")
        logger.info("=" * 60)
        logger.info("AGGREGATE STATISTICS BY SPEAKER:")
        for speaker in ["Lisa", "Pia"]:
            total_q = aggregate_speaker_stats[speaker]["questions"]
            total_t = aggregate_speaker_stats[speaker]["trials"]
            avg_q = total_q / total_t if total_t > 0 else 0
            logger.info(f"  {speaker}: {total_q} questions across {total_t} trials (avg: {avg_q:.2f} questions/trial)")
        logger.info("=" * 60)
        logger.info("AGGREGATE STATISTICS BY TRIAL TYPE:")
        for trial_type, label in [("fully_spec", "Fully specified"), ("underspec", "Underspecified")]:
            total_q = aggregate_trial_type_stats[trial_type]["questions"]
            total_t = aggregate_trial_type_stats[trial_type]["trials"]
            avg_q = total_q / total_t if total_t > 0 else 0
            logger.info(f"  {label}: {total_q} questions across {total_t} trials (avg: {avg_q:.2f} questions/trial)")
        logger.info("=" * 60)

        try:
            result = EvalResult(
                accuracy=overall_accuracy,
                avg_questions_per_instruction=overall_avg_questions,
                overall_avg_score=overall_avg_score
            )
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=result.model_dump_json())),
                ],
                name="Result",
            )
            
            # NEW: Save final results with metadata
            final_results = {
                "model_version": self._model_version,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "overall_accuracy": overall_accuracy,
                "overall_avg_questions": overall_avg_questions,
                "overall_avg_score": overall_avg_score,
                "aggregate_stats_by_speaker": aggregate_speaker_stats,
                "aggregate_stats_by_trial_type": aggregate_trial_type_stats,
                "individual_seeds": all_results
            }
            
            final_file = self._checkpoint_dir / f"{model_name_clean}_final_{run_id}.json"
            final_file.write_text(json.dumps(final_results, indent=2))
            logger.info(f"Saved final results to {final_file}")
            
            # Also add detailed results as artifact (optional)
            # await updater.add_artifact(
            #     parts=[
            #         Part(root=TextPart(text=json.dumps(final_results, indent=2))),
            #     ],
            #     name="Detailed_Results",
            # )
        finally:
            self._tool_provider.reset()
        return result

    def _save_checkpoint(self, checkpoint_file: Path, results: dict, run_id: str) -> None:
        """Save checkpoint with model version and metadata."""
        checkpoint_data = {
            "model_version": self._model_version,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "completed_seeds": len(results),
            "results": results
        }
        
        try:
            checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
            logger.info(f"Checkpoint saved: {checkpoint_file} (completed {len(results)} seeds)")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        return True, "ok"

    async def eval_message(self, response: str, target_structure: str):
        # Strip whitespace and check if response starts with valid action
        response = response.strip()
        
        # Check if response starts with [BUILD] or [ASK]
        if not (response.startswith("[BUILD]") or response.startswith("[ASK]")):
            # Invalid response - treat as incorrect structure and move on
            points = -10
            logger.warning(f"Invalid response format (not [BUILD] or [ASK]): {response[:100]}")
            return {
                "message": f"Incorrect structure. {points} points. Expected [BUILD] or [ASK], but got: {response[:50]}... Moving to next instruction.",
                "num_correct": 0,
                "num_questions": 0,
                "built": True,  # Move on to next instruction
                "points": points
            }
        
        string_response = response.split(";")
        action = string_response[0]
        
        match action:
            case "[BUILD]":
                content = self._normalize_structure(string_response[1:])
                target_structure_set = self._normalize_structure(target_structure.split(";"))
                if content == target_structure_set:
                    points = 10
                    return {"message": f"Correct structure built! +{points} points. {target_structure}",
                            "num_correct": 1,
                            "num_questions": 0,
                            "built": True,
                            "points": points
                            }
                else:
                    points = -10
                    return {"message": f"Incorrect structure. {points} points. Expected: {target_structure}, but got: {';'.join(content)}",
                            "num_correct": 0,
                            "num_questions": 0,
                            "built": True,
                            "points": points
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
                points = -5
                return {"message": f"Answer: {answer} ({points} points for asking)",
                        "num_correct": None,
                        "num_questions": 1,
                        "built": False,
                        "points": points}

            case _:
                # Fallback case (should not reach here due to check above, but keeping for safety)
                points = -10
                return {"message": f"Invalid response format. {points} points. Expected [BUILD] or [ASK]. Moving to next instruction.",
                        "num_correct": 0,
                        "num_questions": 0,
                        "built": True,
                        "points": points
                        }


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
