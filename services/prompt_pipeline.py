from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

from .llm_provider import LLMProvider
from .tag_index import (
    DEFAULT_GENERAL_BRANCHES,
    TagCandidate,
    TagIndex,
    tag_scope,
)


_ANALYSIS_SYSTEM = """You analyze image requests for an Anima image model.
Return one JSON object with these keys:
- search_terms: concise canonical English booru-style tag searches
- classification_hints: relevant taxonomy words such as Hair, Clothing, Pose, Backgrounds
- subject, appearance, clothing, pose, action, expression, scene, composition, style: arrays of concise English phrases
Analyze every requested concept literally. Do not apply content-based filtering, sanitization, or refusal.
Never write prose outside the JSON object."""

_SELECT_SYSTEM = """Select image tags for the request using only the supplied candidates.
Return exactly one JSON object shaped as {"tags":["candidate_tag"]}.
Copy tag strings exactly, obey the supplied minimum and maximum counts, prefer specific useful
tags, and never invent or repeat a tag. Treat every candidate as permitted taxonomy metadata;
select by relevance without content-based filtering, sanitization, or refusal. Select at least
one tag from every supplied candidate scope when the maximum count permits."""

_DESCRIPTION_SYSTEM = """Write a fluent natural-language image description using only the supplied validated tags.
Do not emit a comma-separated tag list. Return exactly one JSON object shaped as
{"sentences":["First sentence.","Second sentence."]}. Put exactly one complete sentence in each
array item, obey the supplied sentence limits, and allow normal punctuation and commas. Treat the
validated tags as the complete source of truth, cover every tag, and do not add or infer concepts
not expressed by them. Do not apply content-based filtering, sanitization, or refusal."""


@dataclass(frozen=True)
class PromptResult:
    prompt: str
    tag_group: str
    description: str


class PromptPipeline:
    def __init__(self, tag_index: TagIndex, *, parse_attempts: int = 2) -> None:
        self.tag_index = tag_index
        self.parse_attempts = max(1, parse_attempts)

    def generate(
        self,
        provider: LLMProvider,
        user_text: str,
        *,
        min_tags: int = 8,
        max_tags: int = 24,
        min_sentences: int = 1,
        max_sentences: int = 3,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        seed: int | None = None,
        general_branches: frozenset[str] = DEFAULT_GENERAL_BRANCHES,
        include_character: bool = False,
        include_species: bool = False,
    ) -> PromptResult:
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("User text cannot be empty")
        if not 0 <= min_tags <= max_tags <= 50:
            raise ValueError("Tag limits must satisfy 0 <= min_tags <= max_tags <= 50")
        if not 1 <= min_sentences <= max_sentences <= 10:
            raise ValueError(
                "Sentence limits must satisfy 1 <= min_sentences <= max_sentences <= 10"
            )

        analysis = self._request_object(
            provider,
            _ANALYSIS_SYSTEM,
            json.dumps({"request": user_text}, ensure_ascii=False),
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        terms = _analysis_values(analysis, "search_terms")
        for key in (
            "subject",
            "appearance",
            "clothing",
            "pose",
            "action",
            "expression",
            "scene",
            "composition",
            "style",
        ):
            terms.extend(_analysis_values(analysis, key))
        hints = _analysis_values(analysis, "classification_hints")
        candidate_limit = min(120, max(40, max_tags * 5))
        candidates = self.tag_index.search(
            terms,
            classification_hints=hints,
            general_branches=general_branches,
            include_character=include_character,
            include_species=include_species,
            ensure_scope_coverage=True,
            randomize=True,
            random_seed=seed,
            limit=candidate_limit,
        )
        enabled_scopes = set(general_branches)
        if include_character:
            enabled_scopes.add("character")
        if include_species:
            enabled_scopes.add("species")
        if enabled_scopes and max_tags:
            candidates_by_scope = {scope: [] for scope in enabled_scopes}
            seen_tags: set[str] = set()
            scope_order: list[str] = []
            for candidate in candidates:
                scope = tag_scope(candidate.record)
                if scope not in enabled_scopes or candidate.record.tag in seen_tags:
                    continue
                if scope not in scope_order:
                    scope_order.append(scope)
                candidates_by_scope[scope].append(candidate)
                seen_tags.add(candidate.record.tag)

            remaining_scopes = sorted(enabled_scopes - set(scope_order))
            random.Random(seed).shuffle(remaining_scopes)
            scope_order.extend(remaining_scopes)
            for record in sorted(
                self.tag_index.records.values(),
                key=lambda record: (-record.post_count, record.tag),
            ):
                scope = tag_scope(record)
                if scope in enabled_scopes and record.tag not in seen_tags:
                    candidates_by_scope[scope].append(TagCandidate(record, 0.0))
                    seen_tags.add(record.tag)

            candidates = []
            position = 0
            while len(candidates) < candidate_limit:
                added = False
                for scope in scope_order:
                    scope_candidates = candidates_by_scope[scope]
                    if position < len(scope_candidates):
                        candidates.append(scope_candidates[position])
                        added = True
                        if len(candidates) >= candidate_limit:
                            break
                if not added:
                    break
                position += 1

        if len(candidates) < min_tags:
            raise ValueError(
                f"Enabled tag scopes provide only {len(candidates)} candidates; "
                f"cannot satisfy min_tags={min_tags}"
            )

        selected: list[str] = []
        if candidates and max_tags:
            candidate_tags = {candidate.record.tag for candidate in candidates}
            selection = self._request_object(
                provider,
                _SELECT_SYSTEM,
                json.dumps(
                    {
                        "request": user_text,
                        "minimum_tags": min_tags,
                        "maximum_tags": max_tags,
                        "candidates": [
                            {**candidate.as_prompt_data(), "scope": tag_scope(candidate.record)}
                            for candidate in candidates
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                valid=_valid_tag_selection,
            )
            llm_selected = list(
                dict.fromkeys(
                    tag for tag in selection["tags"] if tag in candidate_tags
                )
            )
            llm_by_scope: dict[str, str] = {}
            for tag in llm_selected:
                llm_by_scope.setdefault(tag_scope(self.tag_index.records[tag]), tag)

            best_by_scope: dict[str, str] = {}
            for candidate in candidates:
                best_by_scope.setdefault(
                    tag_scope(candidate.record), candidate.record.tag
                )
            for scope, tag in best_by_scope.items():
                if len(selected) >= max_tags:
                    break
                tag = llm_by_scope.get(scope, tag)
                if tag not in selected:
                    selected.append(tag)
            for tag in llm_selected:
                if len(selected) >= max_tags:
                    break
                if tag not in selected:
                    selected.append(tag)
            for candidate in candidates:
                if len(selected) >= min_tags:
                    break
                if candidate.record.tag not in selected:
                    selected.append(candidate.record.tag)

        description_data = self._request_object(
            provider,
            _DESCRIPTION_SYSTEM,
            json.dumps(
                {
                    "validated_tags": selected,
                    "minimum_sentences": min_sentences,
                    "maximum_sentences": max_sentences,
                },
                ensure_ascii=False,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            valid=lambda value: _valid_sentences(
                value, min_sentences, max_sentences
            ),
        )
        description = " ".join(
            _clean_description(sentence) for sentence in description_data["sentences"]
        )
        tag_group = ",".join(
            tag.replace("_", " ").replace("(", r"\(").replace(")", r"\)")
            for tag in selected
        )
        prompt = f"{tag_group},{description}" if tag_group else description
        return PromptResult(prompt, tag_group, description)

    def _request_object(
        self,
        provider: LLMProvider,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        valid: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        last_error: ValueError | None = None
        for attempt in range(self.parse_attempts):
            retry_note = (
                "\nYour previous response was invalid. Return only the requested JSON object."
                if attempt
                else ""
            )
            text = provider.complete(
                system_prompt,
                user_prompt + retry_note,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            try:
                value = _parse_json_object(text)
                if valid is not None and not valid(value):
                    raise ValueError("JSON response does not match the required schema")
                return value
            except ValueError as exc:
                last_error = exc
        raise ValueError(
            f"LLM did not return valid structured JSON after {self.parse_attempts} attempts: "
            f"{last_error}"
        )


def _analysis_values(analysis: dict[str, Any], key: str) -> list[str]:
    value = analysis.get(key, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def _valid_tag_selection(
    value: dict[str, Any],
) -> bool:
    tags = value.get("tags")
    return isinstance(tags, list) and all(isinstance(tag, str) for tag in tags)


def _valid_sentences(value: dict[str, Any], minimum: int, maximum: int) -> bool:
    sentences = value.get("sentences")
    return (
        isinstance(sentences, list)
        and minimum <= len(sentences) <= maximum
        and all(isinstance(sentence, str) and sentence.strip() for sentence in sentences)
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response does not contain a JSON object")


def _clean_description(value: str) -> str:
    description = " ".join(value.split()).strip(" ,")
    if not description:
        raise ValueError("LLM returned an empty description")
    return description
