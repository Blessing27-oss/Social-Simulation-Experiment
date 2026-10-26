import time
import logging
import litellm

_litellm_completion = litellm.completion

def _completion_with_retry(*args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return _litellm_completion(*args, **kwargs)
        except (litellm.exceptions.ServiceUnavailableError, litellm.exceptions.Timeout) as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logging.warning(f"API error ({type(e).__name__}), retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)

litellm.completion = _completion_with_retry

from agents.base import BaseAgent, AgentConfig


class LLMAgent(BaseAgent):
    """LLM-backed agent using litellm."""
    def __init__(self, config: AgentConfig, model: str = "gpt-4o-mini", response_model: type | None = None) -> None:
        super().__init__(config)
        self.model = model
        self.system_prompt = config.system_prompt
        self.response_model = response_model

    def act(self, observation: str) -> str | dict:
        trace = self.act_with_trace(observation)
        return trace["action"]

    def act_with_trace(self, observation: str) -> dict:
        prompt = self._build_prompt(observation)
        payload = {
            "model": self.model,
            "temperature": self.config.temperature,
            "messages": prompt
        }
        if self.response_model:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "action_schema",
                    "strict": True,
                    "schema": self.response_model.model_json_schema()
                }
            }

        response = litellm.completion(**payload)
        text = response["choices"][0]["message"]["content"].strip()

        action_out = self.response_model.model_validate_json(text).model_dump() if self.response_model else text

        return {"action": action_out, "action_text": text, "messages": prompt}

    def _build_prompt(self, observation: str) -> list[dict]:
        system = self.system_prompt
        persona = self.config.persona or ""
        memory_block = ""

        if self.config.use_memory:
            memories = self.memory.retrieve(observation, k=3)
            if memories:
                memory_block = "\n".join(f"- {m.text}" for m in memories)
                memory_block = f"Relevant memory:\n{memory_block}\n"

        user = f"{persona}\n{memory_block}\nObservation:\n{observation}\n\nRespond with only the action:"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
