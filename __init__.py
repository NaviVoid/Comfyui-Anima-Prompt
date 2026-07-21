from .nodes.llm_loaders import LocalLLMLoader, OpenAILLMLoader
from .nodes.prompt_generator import AnimaPromptGenerator


NODE_CLASS_MAPPINGS = {
    "AnimaLocalLLMLoader": LocalLLMLoader,
    "AnimaOpenAILLMLoader": OpenAILLMLoader,
    "AnimaPromptGenerator": AnimaPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLocalLLMLoader": "Anima Local LLM Loader",
    "AnimaOpenAILLMLoader": "Anima OpenAI LLM Loader",
    "AnimaPromptGenerator": "Anima Prompt Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
