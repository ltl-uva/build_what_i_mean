from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class QAClient(Protocol):
    async def answer(self, *, question: str, target_structure: str) -> str:
        ...


class DummyQuestionAnswerer:
    async def answer(self, *, question: str, target_structure: str) -> str:
        return "Yellow"


@dataclass
class QuestionAnswerer:
    model: str
    api_key: str
    base_url: Optional[str] = None
    timeout: float = 30.0
    temperature: float = 0.2
    max_tokens: int = 256

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def from_env(cls) -> Optional[QAClient]:
        mode = os.getenv("AGENT_QA_MODE", "openai").strip().lower()
        if mode == "dummy":
            return DummyQuestionAnswerer()
        if mode != "openai":
            return None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "256"))
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def answer(self, *, question: str, target_structure: str) -> str:
        system_prompt = (
            "You answer questions about a target block structure. "
            "Use only the target structure provided. "
            "Respond with a concise answer only."
        )
        user_prompt = (
            f"Target structure:\n{target_structure}\n\n"
            f"Question: {question}"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning("OpenAI QA failed: %s", exc)
            return "Unable to answer the question right now."

        choice = response.choices[0].message
        content = (choice.content or "").strip()
        return content or "No answer."
