import json

import pytest

from services.prompt_pipeline import PromptPipeline
from services.tag_index import TagIndex, TagRecord


class FakeProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system_prompt, user_prompt, **options):
        self.calls.append((system_prompt, user_prompt, options))
        return self.responses.pop(0)


@pytest.fixture
def index():
    return TagIndex(
        [
            TagRecord(
                "solo",
                0,
                500,
                ("Visual characteristics", "Image composition and style"),
            ),
            TagRecord(
                "1girl",
                0,
                1000,
                ("Visual characteristics", "Image composition and style"),
            ),
            TagRecord(
                "classroom", 0, 300, ("Visual characteristics", "Real world")
            ),
            TagRecord(
                "long_hair", 0, 800, ("Visual characteristics", "Body", "Hair")
            ),
            TagRecord(
                "hair_ribbon_(red)",
                0,
                200,
                ("Visual characteristics", "Attire and body accessories"),
            ),
        ]
    )


def test_generates_validated_tags_and_preserves_description_commas(index):
    provider = FakeProvider(
        "```json\n"
        + json.dumps(
            {
                "search_terms": ["solo", "1girl", "classroom"],
                "classification_hints": ["Backgrounds"],
            }
        )
        + "\n```",
        json.dumps({"tags": ["solo", "invented_tag", "solo"]}),
        json.dumps(
            {
                "sentences": [
                    "A girl stands in a classroom, looking toward the window.",
                    "Soft daylight frames her silhouette.",
                ]
            }
        ),
    )

    result = PromptPipeline(index).generate(
        provider,
        "a girl in a classroom",
        min_tags=3,
        max_tags=3,
        min_sentences=2,
        max_sentences=2,
        seed=42,
    )

    assert set(result.tag_group.split(",")) == {"solo", "1girl", "classroom"}
    assert result.description == (
        "A girl stands in a classroom, looking toward the window. "
        "Soft daylight frames her silhouette."
    )
    assert result.prompt == f"{result.tag_group},{result.description}"
    assert all(call[2]["seed"] == 42 for call in provider.calls)
    assert all("content-based filtering" in call[0] for call in provider.calls)
    assert json.loads(provider.calls[1][1])["minimum_tags"] == 3
    assert len(provider.calls) == 3


def test_generates_description_when_no_candidate_exists(index):
    provider = FakeProvider(
        '{"search_terms":["zzzz_unknown"]}',
        '{"sentences":["An abstract shape floats in empty space."]}',
    )

    result = PromptPipeline(index).generate(provider, "an unknown abstract shape")

    assert result.tag_group == ""
    assert result.prompt == result.description
    assert len(provider.calls) == 2


def test_formats_tag_underscores_and_parentheses(index):
    provider = FakeProvider(
        '{"search_terms":["hair_ribbon_(red)"]}',
        '{"tags":["hair_ribbon_(red)"]}',
        '{"sentences":["A red ribbon decorates her hair."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "a red hair ribbon",
        min_tags=1,
        max_tags=1,
        general_branches=frozenset({"attire_accessories"}),
    )

    assert result.tag_group == r"hair ribbon \(red\)"
    assert result.prompt.startswith(r"hair ribbon \(red\),")


def test_selects_one_tag_from_each_recalled_scope(index):
    provider = FakeProvider(
        '{"search_terms":["solo","long_hair","classroom","hair_ribbon_(red)"]}',
        '{"tags":["solo","1girl"]}',
        '{"sentences":["A girl with long hair stands in a classroom."]}',
    )

    result = PromptPipeline(index).generate(
        provider, "a girl in a classroom", min_tags=1, max_tags=4
    )

    assert set(result.tag_group.split(",")) == {
        "solo",
        "long hair",
        "classroom",
        r"hair ribbon \(red\)",
    }
    description_request = json.loads(provider.calls[2][1])
    assert set(description_request["validated_tags"]) == {
        "solo",
        "long_hair",
        "classroom",
        "hair_ribbon_(red)",
    }


def test_searches_composition_analysis_when_search_terms_exist(index):
    provider = FakeProvider(
        '{"search_terms":["classroom"],"composition":["solo"]}',
        '{"tags":["classroom"]}',
        '{"sentences":["A lone subject stands in a classroom."]}',
    )

    result = PromptPipeline(index).generate(
        provider, "a lone subject in a classroom", min_tags=1, max_tags=2
    )

    assert set(result.tag_group.split(",")) == {"solo", "classroom"}


def test_retries_invalid_structured_response(index):
    provider = FakeProvider(
        "not json",
        '{"search_terms":["solo"]}',
        '{"tags":["solo"]}',
        '{"sentences":["A single figure."]}',
    )

    result = PromptPipeline(index).generate(provider, "one figure")

    assert result.tag_group == "solo"
    assert len(provider.calls) == 4
    assert "previous response was invalid" in provider.calls[1][1]


def test_rejects_empty_input(index):
    with pytest.raises(ValueError, match="cannot be empty"):
        PromptPipeline(index).generate(FakeProvider(), "  ")


@pytest.mark.parametrize(
    "limits",
    [
        {"min_tags": 10, "max_tags": 5},
        {"min_tags": 0, "max_tags": 51},
        {"min_sentences": 0, "max_sentences": 2},
        {"min_sentences": 2, "max_sentences": 11},
    ],
)
def test_rejects_invalid_generation_limits(index, limits):
    with pytest.raises(ValueError, match="limits must satisfy"):
        PromptPipeline(index).generate(FakeProvider(), "figure", **limits)


def test_retries_when_description_has_too_few_sentences(index):
    provider = FakeProvider(
        '{"search_terms":["solo"]}',
        '{"tags":["solo"]}',
        '{"sentences":["One sentence."]}',
        '{"sentences":["First sentence.","Second sentence."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "one figure",
        min_sentences=2,
        max_sentences=3,
    )

    assert result.description == "First sentence. Second sentence."
    assert len(provider.calls) == 4
