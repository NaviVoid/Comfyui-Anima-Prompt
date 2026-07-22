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


def test_comfyui_package_registers_switch_nodes():
    module = load_package()

    assert set(module.NODE_CLASS_MAPPINGS) == {
        "AnimaAdultContentSwitches",
        "AnimaLivingNatureSwitches",
        "AnimaLocalLLMLoader",
        "AnimaObjectsEquipmentSwitches",
        "AnimaOpenAILLMLoader",
        "AnimaPromptGenerator",
        "AnimaScenesActivitiesCultureSwitches",
        "AnimaSubjectAppearanceSwitches",
        "AnimaVisualCompositionSwitches",
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
    assert generator_inputs["required"]["switch_list"] == (
        "ANIMA_TAG_SWITCH_LIST",
    )
    assert "optional" not in generator_inputs

    expected_counts = {
        "AnimaAdultContentSwitches": 19,
        "AnimaLivingNatureSwitches": 3,
        "AnimaObjectsEquipmentSwitches": 29,
        "AnimaScenesActivitiesCultureSwitches": 31,
        "AnimaSubjectAppearanceSwitches": 10,
        "AnimaVisualCompositionSwitches": 13,
    }
    for name, count in expected_counts.items():
        node = module.NODE_CLASS_MAPPINGS[name]
        inputs = node.INPUT_TYPES()
        assert node.RETURN_TYPES == ("ANIMA_TAG_SWITCH_LIST",)
        assert len(inputs["required"]) == count
        assert list(inputs["required"]) == sorted(inputs["required"])
        assert all(value[1]["default"] is True for value in inputs["required"].values())
        assert inputs["optional"] == {
            "switch_list": ("ANIMA_TAG_SWITCH_LIST",)
        }
    assert list(
        module.NODE_CLASS_MAPPINGS["AnimaLivingNatureSwitches"]
        .INPUT_TYPES()["required"]
    ) == ["flowers", "legendary_creatures", "list_of_animals"]


def test_switch_nodes_chain_and_replace_their_group():
    module = load_package()
    living = module.NODE_CLASS_MAPPINGS["AnimaLivingNatureSwitches"]()
    subject = module.NODE_CLASS_MAPPINGS["AnimaSubjectAppearanceSwitches"]()
    living_values = {
        name: name == "list_of_animals"
        for name in living.INPUT_TYPES()["required"]
    }
    subject_values = {
        name: name == "body_parts"
        for name in subject.INPUT_TYPES()["required"]
    }

    first = living.build(**living_values)[0]
    second = subject.build(first, **subject_values)[0]
    replaced = living.build(
        second,
        **{name: False for name in living.INPUT_TYPES()["required"]},
    )[0]

    assert {switch.rsplit(".", 1)[-1] for switch in first} == {"list_of_animals"}
    assert {switch.split(".", 1)[0] for switch in second} == {
        "living_nature",
        "subject_appearance",
    }
    assert {switch.rsplit(".", 1)[-1] for switch in replaced} == {"body_parts"}


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


def test_local_provider_passes_json_schema(monkeypatch):
    captured = {}

    class FakeLlama:
        def __init__(self, **options):
            pass

        def create_chat_completion(self, **options):
            captured.update(options)
            return {"choices": [{"message": {"content": '{"tags":[]}'}}]}

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    module = load_package()
    provider_module = sys.modules[f"{module.__name__}.services.llm_provider"]
    provider = provider_module.LocalLlamaProvider(
        "model.gguf",
        context_length=2048,
        gpu_layers=-1,
        threads=4,
        batch_size=128,
    )
    schema = {"title": "tags", "type": "object"}

    provider.complete(
        "system",
        "user",
        temperature=0.0,
        max_tokens=64,
        response_schema=schema,
    )

    assert captured["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }


def test_openai_provider_passes_strict_json_schema():
    module = load_package()
    provider_module = sys.modules[f"{module.__name__}.services.llm_provider"]
    provider = provider_module.OpenAIProvider(
        "test-model",
        api_key_env="OPENAI_API_KEY",
        api_key="test-key",
        timeout=10.0,
        max_retries=1,
    )
    captured = {}

    def create(**options):
        captured.update(options)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"tags":[]}'))]
        )

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    schema = {"title": "tags", "type": "object"}

    provider.complete(
        "system",
        "user",
        temperature=0.0,
        max_tokens=64,
        response_schema=schema,
    )

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "tags", "strict": True, "schema": schema},
    }
