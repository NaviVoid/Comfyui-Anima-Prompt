from __future__ import annotations

from pathlib import Path

from ..services.prompt_pipeline import PromptPipeline
from ..services.tag_index import load_tag_index


_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tags.csv"
_GENERAL_TOGGLES = dict(sorted({
    "general_actions": "actions",
    "general_attire_accessories": "attire_accessories",
    "general_bdsm_and_torture": "bdsm_and_torture",
    "general_body": "body",
    "general_composition": "composition",
    "general_composition_style": "composition_style",
    "general_creatures": "creatures",
    "general_food": "food",
    "general_games": "games",
    "general_lighting": "lighting",
    "general_misc_objects": "misc_objects",
    "general_perspective_depth": "perspective_depth",
    "general_plants": "plants",
    "general_real_world": "real_world",
    "general_sex_acts": "sex_acts",
    "general_sex_objects": "sex_objects",
    "general_sexual_positions": "sexual_positions",
    "general_vehicles": "vehicles",
    "general_view_angle": "view_angle",
    "general_weapons": "weapons",
}.items()))


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
            "optional": {
                **{
                    name: ("BOOLEAN", {"default": True})
                    for name in _GENERAL_TOGGLES
                },
                "include_character_tags": ("BOOLEAN", {"default": False}),
                "include_species_tags": ("BOOLEAN", {"default": False}),
            },
        }

    def generate(
        self,
        llm,
        user_text: str,
        min_tags: int,
        max_tags: int,
        min_sentences: int,
        max_sentences: int,
        temperature: float,
        max_tokens: int,
        seed: int,
        general_actions: bool = True,
        general_attire_accessories: bool = True,
        general_bdsm_and_torture: bool = True,
        general_body: bool = True,
        general_composition: bool = True,
        general_composition_style: bool = True,
        general_creatures: bool = True,
        general_food: bool = True,
        general_games: bool = True,
        general_lighting: bool = True,
        general_misc_objects: bool = True,
        general_perspective_depth: bool = True,
        general_plants: bool = True,
        general_real_world: bool = True,
        general_sex_acts: bool = True,
        general_sex_objects: bool = True,
        general_sexual_positions: bool = True,
        general_vehicles: bool = True,
        general_view_angle: bool = True,
        general_weapons: bool = True,
        include_character_tags: bool = False,
        include_species_tags: bool = False,
    ):
        index = load_tag_index(_DATA_PATH)
        toggle_values = locals()
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
            general_branches=frozenset(
                branch
                for name, branch in _GENERAL_TOGGLES.items()
                if toggle_values[name]
            ),
            include_character=include_character_tags,
            include_species=include_species_tags,
        )
        return result.prompt, result.tag_group, result.description
