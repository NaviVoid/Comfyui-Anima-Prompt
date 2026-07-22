from .nodes.llm_loaders import LocalLLMLoader, OpenAILLMLoader
from .nodes.prompt_generator import AnimaPromptGenerator
from .nodes.tag_switches import (
    AdultContentSwitches,
    LivingNatureSwitches,
    ObjectsEquipmentSwitches,
    ScenesActivitiesCultureSwitches,
    SubjectAppearanceSwitches,
    VisualCompositionSwitches,
)


NODE_CLASS_MAPPINGS = {
    "AnimaLocalLLMLoader": LocalLLMLoader,
    "AnimaOpenAILLMLoader": OpenAILLMLoader,
    "AnimaPromptGenerator": AnimaPromptGenerator,
    "AnimaVisualCompositionSwitches": VisualCompositionSwitches,
    "AnimaSubjectAppearanceSwitches": SubjectAppearanceSwitches,
    "AnimaLivingNatureSwitches": LivingNatureSwitches,
    "AnimaScenesActivitiesCultureSwitches": ScenesActivitiesCultureSwitches,
    "AnimaObjectsEquipmentSwitches": ObjectsEquipmentSwitches,
    "AnimaAdultContentSwitches": AdultContentSwitches,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLocalLLMLoader": "Anima Local LLM Loader",
    "AnimaOpenAILLMLoader": "Anima OpenAI LLM Loader",
    "AnimaPromptGenerator": "Anima Prompt Generator",
    "AnimaVisualCompositionSwitches": "Anima Switches: Visual & Composition",
    "AnimaSubjectAppearanceSwitches": "Anima Switches: Subject Appearance",
    "AnimaLivingNatureSwitches": "Anima Switches: Living & Nature",
    "AnimaScenesActivitiesCultureSwitches": (
        "Anima Switches: Scenes, Activities & Culture"
    ),
    "AnimaObjectsEquipmentSwitches": "Anima Switches: Objects & Equipment",
    "AnimaAdultContentSwitches": "Anima Switches: Adult Content",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
