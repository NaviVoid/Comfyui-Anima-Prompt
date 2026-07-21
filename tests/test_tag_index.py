import csv
import os

import pytest

from services.tag_index import (
    TagDataError,
    TagIndex,
    TagRecord,
    load_tag_index,
    parse_tag_csv,
)


FIELDS = [
    "tag",
    "category",
    "post_count",
    *(f"classification_{index}" for index in range(1, 8)),
]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row(tag, category=0, post_count=1, *classifications):
    value = {"tag": tag, "category": category, "post_count": post_count}
    value.update(
        {
            f"classification_{index}": classifications[index - 1]
            if index <= len(classifications)
            else ""
            for index in range(1, 8)
        }
    )
    return value


def test_builds_tree_searches_and_caches(tmp_path):
    path = tmp_path / "tags.csv"
    write_csv(
        path,
        [
            row(
                "solo",
                0,
                500,
                "Visual characteristics",
                "Image composition and style",
            ),
            row(
                "1girl",
                0,
                1000,
                "Visual characteristics",
                "Image composition and style",
            ),
            row("long_hair", 0, 900, "Visual characteristics", "Body", "Hair"),
            row("red_hair", 0, 800, "Visual characteristics", "Body", "Hair"),
            row("fox", 12, 700),
        ],
    )

    index = load_tag_index(path)

    assert len(index) == 5
    assert index.classified_count == 4
    assert index.max_classification_depth == 3
    assert "Body" in index.classification_tree["Visual characteristics"]
    assert [item.record.tag for item in index.search(["long hair"], limit=2)] == [
        "long_hair",
        "red_hair",
    ]
    assert load_tag_index(path) is index

    write_csv(path, [row("solo", 0, 500)])
    os.utime(path, None)
    assert load_tag_index(path) is not index


def test_search_enforces_tag_scope():
    index = TagIndex(
        [
            TagRecord("long_hair", 0, 100, ("Visual characteristics", "Body")),
            TagRecord(
                "sword",
                0,
                90,
                ("Visual characteristics", "Objects", "List of weapons"),
            ),
            TagRecord(
                "car",
                0,
                85,
                ("Visual characteristics", "Objects", "List of ground vehicles"),
            ),
            TagRecord(
                "collar",
                0,
                84,
                ("Visual characteristics", "Objects", "Sex objects"),
            ),
            TagRecord(
                "holding_book",
                0,
                83,
                ("Visual characteristics", "Objects", "Holding tags"),
            ),
            TagRecord(
                "cake",
                0,
                82,
                ("Visual characteristics", "More", "Food tags"),
            ),
            TagRecord(
                "looking_down",
                0,
                81,
                ("Visual characteristics", "More", "Verbs and Gerunds"),
            ),
            TagRecord(
                "fire",
                0,
                80,
                ("Visual characteristics", "More", "Fire"),
            ),
            TagRecord(
                "hatsune_miku",
                4,
                80,
                ("Copyrights, artists, projects and media", "Characters"),
            ),
            TagRecord("fox", 12, 70, ("Visual characteristics", "Creatures")),
            TagRecord(
                "school_uniform",
                0,
                60,
                ("Copyrights, artists, projects and media", "More"),
            ),
            TagRecord("translation_note", 0, 50, ("Metatags", "metatags")),
            TagRecord("unclassified", 0, 1000, ()),
        ]
    )
    terms = [
        "long_hair",
        "sword",
        "car",
        "collar",
        "holding_book",
        "cake",
        "looking_down",
        "fire",
        "hatsune_miku",
        "fox",
        "school_uniform",
        "translation_note",
        "unclassified",
    ]

    defaults = {candidate.record.tag for candidate in index.search(terms)}
    body_only = {
        candidate.record.tag
        for candidate in index.search(terms, general_branches={"body"})
    }
    expanded = {
        candidate.record.tag
        for candidate in index.search(
            terms,
            general_branches={"body"},
            include_character=True,
            include_species=True,
        )
    }

    assert defaults == {
        "long_hair",
        "sword",
        "car",
        "collar",
        "holding_book",
        "cake",
        "looking_down",
    }
    assert body_only == {"long_hair"}
    assert expanded == {"long_hair", "hatsune_miku", "fox"}
    assert {
        branch: {candidate.record.tag for candidate in index.search(terms, general_branches={branch})}
        for branch in (
            "weapons",
            "vehicles",
            "sex_objects",
            "misc_objects",
            "food",
            "actions",
        )
    } == {
        "weapons": {"sword"},
        "vehicles": {"car"},
        "sex_objects": {"collar"},
        "misc_objects": {"holding_book"},
        "food": {"cake"},
        "actions": {"looking_down"},
    }
    assert not index.search([], classification_hints=["Characters", "Body"])


def test_search_preserves_recalled_scope_coverage():
    index = TagIndex(
        [
            TagRecord("hair", 0, 1000, ("Visual characteristics", "Body")),
            TagRecord("red_hair", 0, 900, ("Visual characteristics", "Body")),
            TagRecord(
                "hair_salon", 0, 1, ("Visual characteristics", "Real world")
            ),
        ]
    )

    assert {
        candidate.record.tag
        for candidate in index.search(
            ["hair"], ensure_scope_coverage=True, limit=2
        )
    } == {"hair", "hair_salon"}

    random_firsts = {
        index.search(["hair"], randomize=True, random_seed=seed, limit=1)[0].record.tag
        for seed in range(10)
    }
    assert len(random_firsts) > 1
    assert index.search(["hair"], randomize=True, random_seed=3, limit=2) == index.search(
        ["hair"], randomize=True, random_seed=3, limit=2
    )


def test_search_excludes_artistic_license_and_year_tags():
    index = TagIndex(
        [
            TagRecord(
                "simple_background",
                0,
                100,
                (
                    "Visual characteristics",
                    "Image composition and style",
                    "Image composition",
                ),
            ),
            TagRecord(
                "alternate_costume",
                0,
                90,
                (
                    "Visual characteristics",
                    "Image composition and style",
                    "Artistic license",
                ),
            ),
            TagRecord(
                "2024",
                0,
                80,
                (
                    "Visual characteristics",
                    "Image composition and style",
                    "Year tags",
                ),
            ),
        ]
    )

    assert [
        candidate.record.tag
        for candidate in index.search(
            ["simple_background", "alternate_costume", "2024"]
        )
    ] == ["simple_background"]


@pytest.mark.parametrize(
    ("bad_row", "message"),
    [
        (row("bad", 1), "unsupported category"),
        (row("bad", 0, -1), "post_count cannot be negative"),
        (row("bad", 0, "many"), "post_count must be an integer"),
    ],
)
def test_reports_invalid_rows(tmp_path, bad_row, message):
    path = tmp_path / "tags.csv"
    write_csv(path, [bad_row])

    with pytest.raises(TagDataError, match=f"Line 2: {message}"):
        parse_tag_csv(path)


def test_rejects_duplicate_tags(tmp_path):
    path = tmp_path / "tags.csv"
    write_csv(path, [row("solo"), row("solo")])

    with pytest.raises(TagDataError, match="Line 3: duplicate tag"):
        parse_tag_csv(path)


def test_rejects_incompatible_header(tmp_path):
    path = tmp_path / "tags.csv"
    fields = [*FIELDS, "aliases"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**row("solo"), "aliases": "alone"})

    with pytest.raises(TagDataError, match="incompatible header.*unexpected: aliases"):
        parse_tag_csv(path)


def test_rejects_incomplete_rows(tmp_path):
    path = tmp_path / "tags.csv"
    path.write_text(",".join(FIELDS) + "\nsolo,0,1\n", encoding="utf-8")

    with pytest.raises(TagDataError, match="Line 2: missing CSV fields"):
        parse_tag_csv(path)


def test_rejects_csv_without_data(tmp_path):
    path = tmp_path / "tags.csv"
    write_csv(path, [])

    with pytest.raises(TagDataError, match="does not contain any data rows"):
        parse_tag_csv(path)


def test_rejects_malformed_csv(tmp_path):
    path = tmp_path / "tags.csv"
    path.write_text(",".join(FIELDS) + '\n"unterminated\n', encoding="utf-8")

    with pytest.raises(TagDataError, match="Malformed CSV near line 2"):
        parse_tag_csv(path)
