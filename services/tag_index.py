from __future__ import annotations

import csv
import math
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CLASSIFICATION_FIELDS = tuple(f"classification_{index}" for index in range(1, 8))
REQUIRED_FIELDS = ("tag", "category", "post_count", *CLASSIFICATION_FIELDS)
CATEGORY_NAMES = {
    0: "General",
    4: "Character",
    7: "General",
    11: "Character",
    12: "Species",
}
GENERAL_BRANCH_PATHS = {
    "attire_accessories": (
        ("Visual characteristics", "Attire and body accessories"),
    ),
    "body": (("Visual characteristics", "Body"),),
    "creatures": (("Visual characteristics", "Creatures"),),
    "games": (("Visual characteristics", "Games"),),
    # Promoted composition paths must precede their broader parent path.
    "view_angle": (
        (
            "Visual characteristics",
            "Image composition and style",
            "Image composition",
            "View Angle",
        ),
    ),
    "composition": (
        (
            "Visual characteristics",
            "Image composition and style",
            "Image composition",
            "Composition",
        ),
    ),
    "lighting": (
        (
            "Visual characteristics",
            "Image composition and style",
            "Image composition",
            "Lighting",
        ),
    ),
    "perspective_depth": (
        (
            "Visual characteristics",
            "Image composition and style",
            "Image composition",
            "Perspective/Depth",
        ),
    ),
    "composition_style": (
        ("Visual characteristics", "Image composition and style"),
    ),
    "weapons": (
        ("Visual characteristics", "Objects", "List of weapons"),
    ),
    "vehicles": (
        ("Visual characteristics", "Objects", "List of ground vehicles"),
        ("Visual characteristics", "Objects", "List of airplanes"),
        ("Visual characteristics", "Objects", "List of ships"),
        ("Visual characteristics", "Objects", "List of helicopters"),
    ),
    "sex_objects": (
        ("Visual characteristics", "Objects", "Sex objects"),
    ),
    "misc_objects": tuple(
        ("Visual characteristics", "Objects", name)
        for name in (
            "Holding tags",
            "Audio tags",
            "List of armor",
            "Cards",
            "List of Pokemon objects",
            "Piercings",
            "Computer",
            "Doors and Gates",
        )
    ),
    "food": (("Visual characteristics", "More", "Food tags"),),
    "actions": (
        ("Visual characteristics", "More", "Verbs and Gerunds"),
    ),
    "plants": (("Visual characteristics", "Plants"),),
    "real_world": (("Visual characteristics", "Real world"),),
    "sex": (("Visual characteristics", "Sex"),),
}
DEFAULT_GENERAL_BRANCHES = frozenset(GENERAL_BRANCH_PATHS)
GENERAL_CATEGORIES = frozenset({0, 7})
CHARACTER_CATEGORIES = frozenset({4, 11})
SPECIES_CATEGORIES = frozenset({12})
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


class TagDataError(ValueError):
    pass


@dataclass(frozen=True)
class TagRecord:
    tag: str
    category: int
    post_count: int
    classifications: tuple[str, ...]

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES[self.category]


def tag_scope(record: TagRecord) -> str | None:
    if not record.classifications:
        return None
    if record.classifications[:3] in (
        ("Visual characteristics", "Image composition and style", "Artistic license"),
        ("Visual characteristics", "Image composition and style", "Year tags"),
    ):
        return None
    if record.category in CHARACTER_CATEGORIES:
        return "character"
    if record.category in SPECIES_CATEGORIES:
        return "species"
    return next(
        (
            name
            for name, paths in GENERAL_BRANCH_PATHS.items()
            if any(record.classifications[: len(path)] == path for path in paths)
        ),
        None,
    )


@dataclass(frozen=True)
class TagCandidate:
    record: TagRecord
    score: float

    def as_prompt_data(self) -> dict[str, object]:
        data: dict[str, object] = {
            "tag": self.record.tag,
            "category": self.record.category_name,
            "post_count": self.record.post_count,
        }
        if self.record.classifications:
            data["classification"] = list(self.record.classifications)
        return data


def normalize_term(value: str) -> str:
    return _NON_WORD.sub("_", value.casefold()).strip("_")


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in normalize_term(value).split("_") if token)


class TagIndex:
    def __init__(self, records: Iterable[TagRecord]) -> None:
        self.records: dict[str, TagRecord] = {}
        self.classification_tree: dict[str, dict] = {}
        self._normalized: dict[str, TagRecord] = {}
        self._token_index: dict[str, list[TagRecord]] = {}
        self._classification_index: dict[str, list[TagRecord]] = {}
        self.classified_count = 0
        self.max_classification_depth = 0

        for record in records:
            self.records[record.tag] = record
            if not record.classifications:
                continue

            normalized = normalize_term(record.tag)
            self._normalized[normalized] = record
            for token in set(_tokens(record.tag)):
                self._token_index.setdefault(token, []).append(record)

            self.classified_count += 1
            self.max_classification_depth = max(
                self.max_classification_depth, len(record.classifications)
            )
            branch = self.classification_tree
            classification_terms: set[str] = set()
            for part in record.classifications:
                branch = branch.setdefault(part, {})
                normalized_part = normalize_term(part)
                classification_terms.add(normalized_part)
                classification_terms.update(_tokens(part))
            for term in classification_terms:
                self._classification_index.setdefault(term, []).append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, tag: str) -> bool:
        return tag in self.records

    def search(
        self,
        terms: Iterable[str],
        *,
        classification_hints: Iterable[str] = (),
        general_branches: Iterable[str] = DEFAULT_GENERAL_BRANCHES,
        include_character: bool = False,
        include_species: bool = False,
        ensure_scope_coverage: bool = False,
        randomize: bool = False,
        random_seed: int | None = None,
        limit: int = 80,
    ) -> list[TagCandidate]:
        if limit <= 0:
            return []

        allowed_general = frozenset(general_branches)
        unknown_branches = allowed_general - DEFAULT_GENERAL_BRANCHES
        if unknown_branches:
            raise ValueError(
                f"Unknown General branches: {', '.join(sorted(unknown_branches))}"
            )

        def allowed(record: TagRecord) -> bool:
            scope = tag_scope(record)
            if scope == "character":
                return include_character
            if scope == "species":
                return include_species
            return scope in allowed_general

        query_terms = list(
            dict.fromkeys(normalize_term(term) for term in terms if normalize_term(term))
        )[:40]
        scores: dict[str, float] = {}

        for term in query_terms:
            exact = self._normalized.get(term)
            if exact is not None and allowed(exact):
                scores[exact.tag] = scores.get(exact.tag, 0.0) + 1000.0

            postings: set[TagRecord] = set()
            for token in _tokens(term):
                postings.update(self._token_index.get(token, ()))
            if not postings and len(term) >= 3:
                postings.update(
                    record
                    for normalized, record in self._normalized.items()
                    if term in normalized and allowed(record)
                )
            for record in postings:
                if not allowed(record):
                    continue
                tag_term = normalize_term(record.tag)
                overlap = len(set(_tokens(term)) & set(_tokens(record.tag)))
                score = 30.0 * overlap
                if term in tag_term:
                    score += 20.0
                scores[record.tag] = scores.get(record.tag, 0.0) + score

        for hint in classification_hints:
            hint_terms = {normalize_term(hint), *_tokens(hint)}
            for hint_term in hint_terms:
                if not hint_term:
                    continue
                for record in self._classification_index.get(hint_term, ()):
                    if record.tag in scores:
                        scores[record.tag] += 5.0

        ranked = sorted(
            (
                TagCandidate(
                    self.records[tag],
                    score + math.log1p(self.records[tag].post_count) / 10.0,
                )
                for tag, score in scores.items()
            ),
            key=lambda candidate: (
                -candidate.score,
                -candidate.record.post_count,
                candidate.record.tag,
            ),
        )
        if randomize:
            random.Random(random_seed).shuffle(ranked)
        if not ensure_scope_coverage:
            return ranked[:limit]

        covered: list[TagCandidate] = []
        covered_scopes: set[str] = set()
        for candidate in ranked:
            scope = tag_scope(candidate.record)
            if scope is not None and scope not in covered_scopes:
                covered.append(candidate)
                covered_scopes.add(scope)
        covered_tags = {candidate.record.tag for candidate in covered}
        return (
            covered
            + [
                candidate
                for candidate in ranked
                if candidate.record.tag not in covered_tags
            ]
        )[:limit]


def parse_tag_csv(path: str | Path) -> TagIndex:
    csv_path = Path(path)
    try:
        handle = csv_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise TagDataError(f"Unable to open tag CSV {str(csv_path)!r}: {exc}") from exc

    records: list[TagRecord] = []
    seen: set[str] = set()
    try:
        with handle:
            reader = csv.DictReader(handle, strict=True)
            fields = reader.fieldnames
            if not fields:
                raise TagDataError("Tag CSV does not contain a header")
            duplicates = sorted(field for field in set(fields) if fields.count(field) > 1)
            missing = [field for field in REQUIRED_FIELDS if field not in fields]
            unexpected = [field for field in fields if field not in REQUIRED_FIELDS]
            if duplicates or missing or unexpected:
                details = []
                if missing:
                    details.append(f"missing: {', '.join(missing)}")
                if unexpected:
                    details.append(f"unexpected: {', '.join(unexpected)}")
                if duplicates:
                    details.append(f"duplicate: {', '.join(duplicates)}")
                raise TagDataError(f"Tag CSV has an incompatible header ({'; '.join(details)})")

            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise TagDataError(f"Line {line_number}: too many CSV fields")
                incomplete = [field for field in REQUIRED_FIELDS if row[field] is None]
                if incomplete:
                    raise TagDataError(
                        f"Line {line_number}: missing CSV fields: {', '.join(incomplete)}"
                    )
                if any("\0" in row[field] for field in REQUIRED_FIELDS):
                    raise TagDataError(f"Line {line_number}: NUL bytes are not allowed")

                tag = row["tag"].strip()
                if not tag:
                    raise TagDataError(f"Line {line_number}: tag cannot be empty")
                if tag in seen:
                    raise TagDataError(f"Line {line_number}: duplicate tag {tag!r}")

                category = _parse_integer(row["category"], "category", line_number)
                if category not in CATEGORY_NAMES:
                    raise TagDataError(
                        f"Line {line_number}: unsupported category {category}; "
                        f"expected one of {sorted(CATEGORY_NAMES)}"
                    )
                post_count = _parse_integer(row["post_count"], "post_count", line_number)
                if post_count < 0:
                    raise TagDataError(f"Line {line_number}: post_count cannot be negative")

                classifications = tuple(
                    row[field].strip() for field in CLASSIFICATION_FIELDS
                )
                first_empty = next(
                    (index for index, value in enumerate(classifications) if not value),
                    len(classifications),
                )
                if any(classifications[first_empty + 1 :]):
                    raise TagDataError(
                        f"Line {line_number}: classification levels must be contiguous"
                    )

                seen.add(tag)
                records.append(
                    TagRecord(tag, category, post_count, classifications[:first_empty])
                )
    except UnicodeDecodeError as exc:
        raise TagDataError(f"Tag CSV must be valid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise TagDataError(
            f"Malformed CSV near line {max(2, reader.line_num)}: {exc}"
        ) from exc

    if not records:
        raise TagDataError("Tag CSV does not contain any data rows")
    return TagIndex(records)


def _parse_integer(value: str | None, field: str, line_number: int) -> int:
    try:
        return int(value or "")
    except ValueError as exc:
        raise TagDataError(f"Line {line_number}: {field} must be an integer") from exc


_CACHE: dict[Path, tuple[int, int, TagIndex]] = {}
_CACHE_LOCK = threading.Lock()


def load_tag_index(path: str | Path) -> TagIndex:
    csv_path = Path(path).resolve()
    try:
        stat = csv_path.stat()
    except OSError as exc:
        raise TagDataError(f"Unable to stat tag CSV {str(csv_path)!r}: {exc}") from exc
    signature = (stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        cached = _CACHE.get(csv_path)
        if cached is not None and cached[:2] == signature:
            return cached[2]
        index = parse_tag_csv(csv_path)
        _CACHE[csv_path] = (*signature, index)
        return index
