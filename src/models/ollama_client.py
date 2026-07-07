"""
Thin wrapper around the Ollama Python client. All model-calling logic
lives here so the rest of the pipeline never needs to know HOW a
prompt reaches a model -- only that calling .generate(prompt) or
.chat(messages) returns text.
"""

import ollama


class OllamaClient:
    """A single local Ollama model, wrapped for reuse across the pipeline."""

    def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """Send a single-turn prompt to the model, return its text reply."""
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list) -> str:
        """
        Send a full multi-turn conversation (a list of {"role", "content"}
        dicts) to the model, return only the newest reply. Needed for
        self-verification: the model must see its own prior answer as
        part of the conversation, not a fresh standalone prompt.
        """
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]
