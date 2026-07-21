from __future__ import annotations

import os
import re
import weakref
from pathlib import Path

from ..services.llm_provider import LocalLlamaProvider, OpenAIProvider


_NO_MODELS = "(No GGUF models found in ComfyUI/models/LLM)"
_LOCAL_CACHE: weakref.WeakValueDictionary[tuple[object, ...], LocalLlamaProvider] = (
    weakref.WeakValueDictionary()
)


def _models_dir() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "LLM"
    except ImportError:
        return Path(__file__).resolve().parents[3] / "models" / "LLM"


def _model_names() -> list[str]:
    root = _models_dir()
    if not root.is_dir():
        return [_NO_MODELS]
    names = sorted(str(path.relative_to(root)) for path in root.rglob("*.gguf"))
    return names or [_NO_MODELS]


def _model_path(model_name: str) -> Path:
    root = _models_dir().resolve()
    path = (root / model_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("GGUF model path must stay inside ComfyUI/models/LLM") from exc
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"GGUF model not found: {model_name}")
    return path


class LocalLLMLoader:
    CATEGORY = "Anima Prompt"
    FUNCTION = "load"
    RETURN_TYPES = ("ANIMA_LLM",)
    RETURN_NAMES = ("llm",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_names(),),
                "context_length": (
                    "INT",
                    {"default": 8192, "min": 512, "max": 131072, "step": 512},
                ),
                "gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 1000, "step": 1},
                ),
                "threads": (
                    "INT",
                    {"default": max(1, (os.cpu_count() or 2) // 2), "min": 1, "max": 256},
                ),
                "batch_size": (
                    "INT",
                    {"default": 512, "min": 32, "max": 4096, "step": 32},
                ),
            }
        }

    def load(
        self,
        model_name: str,
        context_length: int,
        gpu_layers: int,
        threads: int,
        batch_size: int,
    ):
        path = _model_path(model_name)
        key = (str(path), context_length, gpu_layers, threads, batch_size)
        provider = _LOCAL_CACHE.get(key)
        if provider is None:
            provider = LocalLlamaProvider(
                str(path),
                context_length=context_length,
                gpu_layers=gpu_layers,
                threads=threads,
                batch_size=batch_size,
            )
            _LOCAL_CACHE[key] = provider
        self._provider = provider
        return (provider,)


class OpenAILLMLoader:
    CATEGORY = "Anima Prompt"
    FUNCTION = "load"
    RETURN_TYPES = ("ANIMA_LLM",)
    RETURN_NAMES = ("llm",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gpt-4.1-mini"}),
                "api_key_env": ("STRING", {"default": "OPENAI_API_KEY"}),
                "timeout": (
                    "FLOAT",
                    {"default": 60.0, "min": 1.0, "max": 600.0, "step": 1.0},
                ),
                "max_retries": ("INT", {"default": 2, "min": 0, "max": 10}),
            },
            "optional": {
                "OPENAI_API_KEY": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
            },
        }

    def load(
        self,
        model_name: str,
        api_key_env: str,
        timeout: float,
        max_retries: int,
        OPENAI_API_KEY: str = "",
    ):
        model_name = model_name.strip()
        api_key_env = api_key_env.strip()
        if not model_name:
            raise ValueError("OpenAI model name cannot be empty")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("API key environment variable name is invalid")
        return (
            OpenAIProvider(
                model_name,
                api_key_env=api_key_env,
                api_key=OPENAI_API_KEY.strip() or None,
                timeout=timeout,
                max_retries=max_retries,
            ),
        )
