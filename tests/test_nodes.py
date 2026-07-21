import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


def load_package():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "anima_prompt_test_package",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_comfyui_package_registers_three_nodes():
    module = load_package()

    assert set(module.NODE_CLASS_MAPPINGS) == {
        "AnimaLocalLLMLoader",
        "AnimaOpenAILLMLoader",
        "AnimaPromptGenerator",
    }
    assert module.NODE_CLASS_MAPPINGS["AnimaLocalLLMLoader"].RETURN_TYPES == (
        "ANIMA_LLM",
    )
    assert module.NODE_CLASS_MAPPINGS["AnimaOpenAILLMLoader"].RETURN_TYPES == (
        "ANIMA_LLM",
    )
    generator_inputs = module.NODE_CLASS_MAPPINGS["AnimaPromptGenerator"].INPUT_TYPES()
    assert generator_inputs["required"]["min_tags"][1]["default"] == 8
    assert generator_inputs["required"]["max_tags"][1]["max"] == 50
    assert generator_inputs["required"]["min_sentences"][1]["min"] == 1
    assert generator_inputs["required"]["max_sentences"][1]["max"] == 10
    assert generator_inputs["optional"]["general_body"][1]["default"] is True
    assert "general_objects" not in generator_inputs["optional"]
    assert "general_misc" not in generator_inputs["optional"]
    assert generator_inputs["optional"]["general_weapons"][1]["default"] is True
    assert generator_inputs["optional"]["general_food"][1]["default"] is True
    assert "general_media_taxonomy" not in generator_inputs["optional"]
    assert "general_metatags" not in generator_inputs["optional"]
    assert generator_inputs["optional"]["include_character_tags"][1]["default"] is False
    assert generator_inputs["optional"]["include_species_tags"][1]["default"] is False


def test_openai_loader_is_lazy():
    OpenAILLMLoader = load_package().NODE_CLASS_MAPPINGS["AnimaOpenAILLMLoader"]

    provider = OpenAILLMLoader().load("test-model", "OPENAI_API_KEY", 10.0, 1)[0]

    assert provider._client is None


def test_local_loader_reuses_same_configuration(monkeypatch):
    module = load_package()
    loader_module = sys.modules[f"{module.__name__}.nodes.llm_loaders"]

    class FakeLocalProvider:
        def __init__(self, model_path, **options):
            self.model_path = model_path
            self.options = options

    monkeypatch.setattr(loader_module, "LocalLlamaProvider", FakeLocalProvider)
    monkeypatch.setattr(loader_module, "_model_path", lambda _: Path("model.gguf"))
    loader_module._LOCAL_CACHE.clear()
    loader_class = module.NODE_CLASS_MAPPINGS["AnimaLocalLLMLoader"]

    first = loader_class().load("model.gguf", 2048, -1, 4, 128)[0]
    second = loader_class().load("model.gguf", 2048, -1, 4, 128)[0]

    assert first is second
    assert "seed" not in first.options


def test_openai_provider_reports_missing_key(monkeypatch):
    module = load_package()
    loader = module.NODE_CLASS_MAPPINGS["AnimaOpenAILLMLoader"]()
    monkeypatch.delenv("ANIMA_TEST_OPENAI_KEY", raising=False)
    provider = loader.load("test-model", "ANIMA_TEST_OPENAI_KEY", 10.0, 1)[0]

    with pytest.raises(RuntimeError, match="fill OPENAI_API_KEY.*ANIMA_TEST_OPENAI_KEY"):
        provider.complete("system", "user", temperature=0.1, max_tokens=10)


def test_openai_loader_accepts_direct_key(monkeypatch):
    module = load_package()
    loader = module.NODE_CLASS_MAPPINGS["AnimaOpenAILLMLoader"]()
    monkeypatch.setenv("ANIMA_TEST_OPENAI_KEY", "environment-key")
    provider = loader.load(
        "test-model", "ANIMA_TEST_OPENAI_KEY", 10.0, 1, "direct-key"
    )[0]
    captured = {}

    def fake_openai(**options):
        captured.update(options)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    provider._get_client()

    assert captured["api_key"] == "direct-key"
