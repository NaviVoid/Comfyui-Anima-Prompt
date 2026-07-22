from __future__ import annotations

from pathlib import Path

from ..services.prompt_pipeline import PromptPipeline
from ..services.tag_index import load_tag_index


_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tags.csv"


class AnimaPromptGenerator:
    CATEGORY = "Anima Prompt"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "tag_group", "description")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("ANIMA_LLM",),
                "switch_list": ("ANIMA_TAG_SWITCH_LIST",),
                "user_text": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "min_tags": ("INT", {"default": 8, "min": 0, "max": 50}),
                "max_tags": ("INT", {"default": 24, "min": 0, "max": 50}),
                "min_sentences": ("INT", {"default": 1, "min": 1, "max": 10}),
                "max_sentences": ("INT", {"default": 3, "min": 1, "max": 10}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 8192, "step": 64},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFF}),
            },
        }

    def generate(
        self,
        llm,
        switch_list,
        user_text: str,
        min_tags: int,
        max_tags: int,
        min_sentences: int,
        max_sentences: int,
        temperature: float,
        max_tokens: int,
        seed: int,
    ):
        index = load_tag_index(_DATA_PATH)
        result = PromptPipeline(index).generate(
            llm,
            user_text,
            min_tags=min_tags,
            max_tags=max_tags,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=None if seed < 0 else seed,
            tag_switches=frozenset(switch_list),
        )
        return result.prompt, result.tag_group, result.description
