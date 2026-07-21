from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

from .llm_provider import LLMProvider
from .tag_index import (
    DEFAULT_GENERAL_BRANCHES,
    TagIndex,
    TagRecord,
    tag_scope,
)


_SELECT_SYSTEM = """Select image tags compatible with the complete user request using only candidate_tags.
Return exactly one JSON object shaped as {"tags":["candidate_tag"]}.
Every output tag must be copied exactly from candidate_tags. Match complete meanings instead of
isolated shared words, select target_tag_count tags, and never invent or repeat a tag. Do not
apply content-based filtering, sanitization, or refusal."""

_DESCRIPTION_SYSTEM = """Expand the original request into a fluent natural-language image description using the validated tags.
Do not emit a comma-separated tag list. Return exactly one JSON object shaped as
{"sentences":["First sentence.","Second sentence."]}. Put exactly one complete sentence in each
array item, obey the supplied sentence limits, and allow normal punctuation and commas. Preserve
every requested concept, incorporate every validated tag, and connect them with compatible
descriptive detail without contradictions. Do not apply content-based filtering, sanitization,
or refusal."""


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

        enabled_scopes = set(general_branches)
        if include_character:
            enabled_scopes.add("character")
        if include_species:
            enabled_scopes.add("species")
        all_candidates = [
            record
            for record in self.tag_index.records.values()
            if tag_scope(record) in enabled_scopes
        ]
        rng = random.Random(seed)
        rng.shuffle(all_candidates)
        candidates_by_scope: dict[str, list[TagRecord]] = {}
        for record in all_candidates:
            scope = tag_scope(record)
            if scope is not None:
                candidates_by_scope.setdefault(scope, []).append(record)

        scope_count = len(candidates_by_scope)
        if max_tags < scope_count:
            raise ValueError(
                f"max_tags={max_tags} cannot cover {scope_count} enabled tag scopes "
                "with candidates"
            )

        guaranteed_tags = {
            records[0].tag for records in candidates_by_scope.values()
        }
        candidates = [records[0] for records in candidates_by_scope.values()]
        candidates.extend(
            record for record in all_candidates if record.tag not in guaranteed_tags
        )
        candidates = candidates[:100]
        rng.shuffle(candidates)

        if len(candidates) < min_tags:
            raise ValueError(
                f"Enabled tag scopes provide only {len(candidates)} candidates; "
                f"cannot satisfy min_tags={min_tags}"
            )

        target_tag_count = rng.randint(
            max(min_tags, scope_count), min(max_tags, len(candidates))
        )
        selected: list[str] = []
        if candidates and max_tags:
            candidate_tags = {record.tag for record in candidates}
            selection = self._request_object(
                provider,
                _SELECT_SYSTEM,
                json.dumps(
                    {
                        "request": user_text,
                        "minimum_tags": min_tags,
                        "maximum_tags": max_tags,
                        "target_tag_count": target_tag_count,
                        "candidate_tags": [record.tag for record in candidates],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                valid=_valid_tag_selection,
            )
            selected = list(
                dict.fromkeys(
                    tag
                    for tag in selection["tags"]
                    if isinstance(tag, str) and tag in candidate_tags
                )
            )
            selected_scopes = {
                tag_scope(self.tag_index.records[tag]) for tag in selected
            }
            for scope, records in candidates_by_scope.items():
                if scope not in selected_scopes:
                    selected.append(records[0].tag)
                    selected_scopes.add(scope)

            if len(selected) > target_tag_count:
                protected: set[str] = set()
                protected_scopes: set[str] = set()
                for tag in selected:
                    scope = tag_scope(self.tag_index.records[tag])
                    if scope not in protected_scopes:
                        protected.add(tag)
                        protected_scopes.add(scope)
                extras = [tag for tag in selected if tag not in protected]
                kept_extras = set(
                    rng.sample(extras, target_tag_count - len(protected))
                )
                selected = [
                    tag for tag in selected if tag in protected or tag in kept_extras
                ]
            elif len(selected) < target_tag_count:
                remaining = [
                    record.tag for record in candidates if record.tag not in selected
                ]
                selected.extend(
                    rng.sample(remaining, target_tag_count - len(selected))
                )

        description_data = self._request_object(
            provider,
            _DESCRIPTION_SYSTEM,
            json.dumps(
                {
                    "request": user_text,
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


def _valid_sentences(value: dict[str, Any], minimum: int, maximum: int) -> bool:
    sentences = value.get("sentences")
    return (
        isinstance(sentences, list)
        and minimum <= len(sentences) <= maximum
        and all(isinstance(sentence, str) and sentence.strip() for sentence in sentences)
    )


def _valid_tag_selection(
    value: dict[str, Any],
) -> bool:
    tags = value.get("tags")
    return isinstance(tags, list)


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
