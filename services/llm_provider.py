from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Backend-neutral text completion interface used as ANIMA_LLM."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalLlamaProvider(LLMProvider):
    def __init__(
        self,
        model_path: str,
        *,
        context_length: int,
        gpu_layers: int,
        threads: int,
        batch_size: int,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for the Anima Local LLM Loader"
            ) from exc

        try:
            self._model = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_gpu_layers=gpu_layers,
                n_threads=threads,
                n_batch=batch_size,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to load GGUF model {model_path!r}: {exc}") from exc
        self._lock = threading.Lock()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": response_schema,
            }

        try:
            with self._lock:
                response = self._model.create_chat_completion(**kwargs)
            return _message_text(response)
        except Exception as exc:
            raise RuntimeError(f"Local LLM generation failed: {exc}") from exc

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            self._model = None
            close = getattr(model, "close", None)
            if close is not None:
                close()

    def __del__(self) -> None:
        self.close()


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key_env: str,
        api_key: str | None,
        timeout: float,
        max_retries: int,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Any = None
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is not None:
                return self._client
            api_key = self._api_key or os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    "OpenAI API key is missing; fill OPENAI_API_KEY or set "
                    f"environment variable {self.api_key_env!r}"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The openai package is required for the Anima OpenAI LLM Loader"
                ) from exc
            self._client = OpenAI(
                api_key=api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            return self._client

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.get("title", "anima_response"),
                    "strict": True,
                    "schema": response_schema,
                },
            }
        try:
            response = self._get_client().chat.completions.create(**kwargs)
            return _message_text(response)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc


def _message_text(response: Any) -> str:
    try:
        choice = response["choices"][0] if isinstance(response, dict) else response.choices[0]
        message = choice["message"] if isinstance(choice, dict) else choice.message
        content = message["content"] if isinstance(message, dict) else message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError("LLM returned an invalid response") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM returned empty text")
    return content.strip()
