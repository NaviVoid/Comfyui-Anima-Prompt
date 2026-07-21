import json

import pytest

from services.prompt_pipeline import PromptPipeline
from services.tag_index import TagIndex, TagRecord, tag_scope


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
        json.dumps({"tags": ["solo", "long_hair", "classroom"]}),
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
        general_branches=frozenset(
            {"composition_style", "real_world", "body"}
        ),
    )

    assert set(result.tag_group.split(",")) == {"solo", "long hair", "classroom"}
    assert result.description == (
        "A girl stands in a classroom, looking toward the window. "
        "Soft daylight frames her silhouette."
    )
    assert result.prompt == f"{result.tag_group},{result.description}"
    assert all(call[2]["seed"] == 42 for call in provider.calls)
    assert all("content-based filtering" in call[0] for call in provider.calls)
    selection_request = json.loads(provider.calls[0][1])
    assert selection_request["request"] == "a girl in a classroom"
    assert selection_request["target_tag_count"] == 3
    assert set(selection_request["candidate_tags_by_scope"]) == {
        "composition_style",
        "real_world",
        "body",
    }
    assert provider.calls[0][2]["temperature"] == 0.0
    assert provider.calls[0][2]["max_tokens"] == 100
    selection_schema = provider.calls[0][2]["response_schema"]
    assert selection_schema["title"] == "anima_tag_selection"
    assert selection_schema["properties"]["tags"]["minItems"] == 3
    assert selection_schema["properties"]["tags"]["maxItems"] == 3
    assert provider.calls[1][2]["temperature"] == 0.3
    assert provider.calls[1][2]["max_tokens"] == 256
    description_schema = provider.calls[1][2]["response_schema"]
    assert description_schema["title"] == "anima_description"
    assert description_schema["properties"]["sentences"]["minItems"] == 2
    assert description_schema["properties"]["sentences"]["maxItems"] == 2
    assert description_schema["properties"]["sentences"]["items"]["pattern"]
    assert len(provider.calls) == 2


def test_generates_description_when_no_candidate_exists(index):
    provider = FakeProvider(
        '{"sentences":["An abstract shape floats in empty space."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "an unknown abstract shape",
        min_tags=0,
        general_branches=frozenset(),
    )

    assert result.tag_group == ""
    assert result.prompt == result.description
    assert len(provider.calls) == 1


def test_formats_tag_underscores_and_parentheses(index):
    provider = FakeProvider(
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


def test_description_receives_request_and_final_tags(index):
    provider = FakeProvider(
        '{"tags":["solo","1girl"]}',
        '{"sentences":["A girl with long hair stands in a classroom."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "a girl in a classroom",
        min_tags=2,
        max_tags=2,
        general_branches=frozenset({"composition_style"}),
    )

    assert result.tag_group == "solo,1girl"
    description_request = json.loads(provider.calls[1][1])
    assert description_request["request"] == "a girl in a classroom"
    assert description_request["validated_tags"] == ["solo", "1girl"]
    assert "Expand the original request" in provider.calls[1][0]


def test_selection_receives_complete_request(index):
    provider = FakeProvider(
        '{"tags":["classroom","solo"]}',
        '{"sentences":["A lone subject stands in a classroom."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "a lone subject in a classroom",
        min_tags=2,
        max_tags=2,
        general_branches=frozenset({"composition_style", "real_world"}),
    )

    assert set(result.tag_group.split(",")) == {"solo", "classroom"}
    assert json.loads(provider.calls[0][1])["request"] == (
        "a lone subject in a classroom"
    )


def test_program_repairs_invalid_llm_selection():
    composition = ("Visual characteristics", "Image composition and style")
    index = TagIndex(
        [
            TagRecord("solo", 0, 500, composition),
            TagRecord("1girl", 0, 400, composition),
            TagRecord("full_body", 0, 300, composition),
            TagRecord("simple_background", 0, 200, composition),
            TagRecord("soft_lighting", 0, 100, composition),
            TagRecord(
                "classroom", 0, 90, ("Visual characteristics", "Real world")
            ),
            TagRecord("long_hair", 0, 80, ("Visual characteristics", "Body")),
        ]
    )
    provider = FakeProvider(
        '{"tags":["classroom","classroom","invented","solo","1girl","full_body","simple_background","soft_lighting"]}',
        '{"sentences":["A full-body portrait is set in a classroom."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "a girl",
        min_tags=4,
        max_tags=4,
        seed=42,
        general_branches=frozenset({"composition_style", "real_world", "body"}),
    )

    validated_tags = json.loads(provider.calls[1][1])["validated_tags"]
    assert len(validated_tags) == 4
    assert len(validated_tags) == len(set(validated_tags))
    assert "invented" not in validated_tags
    assert all(tag in index.records for tag in validated_tags)


def test_guarantees_enabled_scope_coverage_before_description():
    index = TagIndex(
        [
            TagRecord("body_a", 0, 6, ("Visual characteristics", "Body")),
            TagRecord("body_b", 0, 5, ("Visual characteristics", "Body")),
            TagRecord(
                "classroom", 0, 4, ("Visual characteristics", "Real world")
            ),
            TagRecord(
                "depth_of_field",
                0,
                3,
                ("Visual characteristics", "Image composition and style"),
            ),
        ]
    )
    scopes = frozenset({"body", "real_world", "composition_style"})
    provider = FakeProvider(
        '{"tags":["body_a","body_b"]}',
        '{"sentences":["A figure stands in a classroom."]}',
    )

    PromptPipeline(index).generate(
        provider,
        "a figure in a classroom",
        min_tags=1,
        max_tags=3,
        general_branches=scopes,
        seed=42,
    )

    candidate_tags_by_scope = json.loads(provider.calls[0][1])[
        "candidate_tags_by_scope"
    ]
    validated_tags = json.loads(provider.calls[1][1])["validated_tags"]
    assert set(candidate_tags_by_scope) == scopes
    assert {tag_scope(index.records[tag]) for tag in validated_tags} == scopes
    with pytest.raises(ValueError, match="max_tags=2 cannot cover 3"):
        PromptPipeline(index).generate(
            FakeProvider(),
            "a figure",
            min_tags=1,
            max_tags=2,
            general_branches=scopes,
        )


def test_limits_random_candidate_pool_to_100():
    index = TagIndex(
        [
            TagRecord(
                f"style_{number}",
                0,
                100 - number,
                ("Visual characteristics", "Image composition and style"),
            )
            for number in range(120)
        ]
    )
    provider = FakeProvider(
        '{"tags":[]}',
        '{"sentences":["A figure is shown."]}',
    )

    PromptPipeline(index).generate(
        provider,
        "a figure",
        min_tags=0,
        max_tags=10,
        seed=42,
        general_branches=frozenset({"composition_style"}),
    )

    candidate_tags_by_scope = json.loads(provider.calls[0][1])[
        "candidate_tags_by_scope"
    ]
    assert sum(map(len, candidate_tags_by_scope.values())) == 100


def test_balances_candidate_pool_across_scopes():
    index = TagIndex(
        [
            TagRecord(
                f"body_{number}", 0, 200 - number, ("Visual characteristics", "Body")
            )
            for number in range(200)
        ]
        + [
            TagRecord(
                f"light_{number}",
                0,
                10 - number,
                (
                    "Visual characteristics",
                    "Image composition and style",
                    "Image composition",
                    "Lighting",
                ),
            )
            for number in range(4)
        ]
    )
    provider = FakeProvider(
        '{"tags":[]}',
        '{"sentences":["A body is shown under deliberate lighting."]}',
    )

    PromptPipeline(index).generate(
        provider,
        "unmatched query xyz",
        min_tags=2,
        max_tags=2,
        seed=42,
        general_branches=frozenset({"body", "lighting"}),
    )

    candidate_tags_by_scope = json.loads(provider.calls[0][1])[
        "candidate_tags_by_scope"
    ]
    assert len(candidate_tags_by_scope["lighting"]) == 4
    assert len(candidate_tags_by_scope["body"]) == 96


def test_prioritizes_explicit_tag_matches_without_losing_scope_balance():
    body = ("Visual characteristics", "Body")
    real_world = ("Visual characteristics", "Real world")
    index = TagIndex(
        [TagRecord("long_hair", 0, 1000, body)]
        + [TagRecord(f"body_{number}", 0, 500 - number, body) for number in range(100)]
        + [TagRecord("library", 0, 1000, real_world)]
        + [
            TagRecord(f"place_{number}", 0, 500 - number, real_world)
            for number in range(100)
        ]
    )
    provider = FakeProvider(
        '{"tags":[]}',
        '{"sentences":["A long-haired reader is in a library."]}',
    )

    PromptPipeline(index).generate(
        provider,
        "long hair in a library",
        min_tags=2,
        max_tags=2,
        seed=42,
        general_branches=frozenset({"body", "real_world"}),
    )

    candidate_tags_by_scope = json.loads(provider.calls[0][1])[
        "candidate_tags_by_scope"
    ]
    assert "long_hair" in candidate_tags_by_scope["body"]
    assert "library" in candidate_tags_by_scope["real_world"]
    assert all(len(tags) >= 25 for tags in candidate_tags_by_scope.values())


def test_randomizes_supplemental_candidates_by_seed():
    index = TagIndex(
        [
            TagRecord(f"body_{number}", 0, 100 - number, ("Visual characteristics", "Body"))
            for number in range(6)
        ]
    )

    def generate(seed):
        provider = FakeProvider(
            '{"tags":[]}',
            '{"sentences":["A body is shown."]}',
        )
        result = PromptPipeline(index).generate(
            provider,
            "a figure",
            min_tags=1,
            max_tags=4,
            seed=seed,
            general_branches=frozenset({"body"}),
        )
        request = json.loads(provider.calls[0][1])
        candidates = tuple(
            (scope, tuple(tags))
            for scope, tags in request["candidate_tags_by_scope"].items()
        )
        return result.tag_group, request["target_tag_count"], candidates

    assert generate(7) == generate(7)
    results = [generate(seed) for seed in range(10)]
    assert all(1 <= target <= 4 for _, target, _ in results)
    assert len({target for _, target, _ in results}) > 1
    assert len({tags for tags, _, _ in results}) > 1


def test_rejects_unachievable_tag_minimum():
    index = TagIndex(
        [
            TagRecord(
                "solo",
                0,
                1,
                ("Visual characteristics", "Image composition and style"),
            )
        ]
    )

    with pytest.raises(ValueError, match="cannot satisfy min_tags=2"):
        PromptPipeline(index).generate(
            FakeProvider(),
            "a figure",
            min_tags=2,
            max_tags=2,
            general_branches=frozenset({"composition_style"}),
        )


def test_retries_invalid_structured_response(index):
    provider = FakeProvider(
        "not json",
        '{"tags":["solo"]}',
        '{"sentences":["A single figure."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "one figure",
        min_tags=1,
        max_tags=1,
        general_branches=frozenset({"composition_style"}),
    )

    assert result.tag_group == "solo"
    assert len(provider.calls) == 3
    assert "previous response was invalid" in provider.calls[1][1]


def test_supplements_invalid_tag_selection(index):
    provider = FakeProvider(
        '{"tags":["invented_tag","solo","solo"]}',
        '{"sentences":["A girl stands alone."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "a girl",
        min_tags=2,
        max_tags=2,
        general_branches=frozenset({"composition_style"}),
    )

    assert result.tag_group == "solo,1girl"
    assert len(provider.calls) == 2


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
        '{"tags":["solo"]}',
        '{"sentences":["One sentence."]}',
        '{"sentences":["First sentence.","Second sentence."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "one figure",
        min_tags=1,
        max_tags=1,
        min_sentences=2,
        max_sentences=3,
        general_branches=frozenset({"composition_style"}),
    )

    assert result.description == "First sentence. Second sentence."
    assert len(provider.calls) == 3


def test_retries_when_description_item_contains_multiple_sentences(index):
    provider = FakeProvider(
        '{"tags":["solo"]}',
        '{"sentences":["First sentence. Extra sentence.","Second sentence."]}',
        '{"sentences":["First sentence.","Second sentence."]}',
    )

    result = PromptPipeline(index).generate(
        provider,
        "one figure",
        min_tags=1,
        max_tags=1,
        min_sentences=2,
        max_sentences=2,
        general_branches=frozenset({"composition_style"}),
    )

    assert result.description == "First sentence. Second sentence."
    assert len(provider.calls) == 3
