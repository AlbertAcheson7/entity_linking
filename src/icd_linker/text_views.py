from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping


SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def unique_texts(values: Iterable[Any]) -> List[str]:
    result, seen = [], set()
    for value in values:
        text = clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def list_field(term: Mapping[str, Any], key: str) -> List[str]:
    value = term.get(key, [])
    if not isinstance(value, list):
        value = [value]
    return unique_texts(value)


def resolve_parent_names(
    term: Mapping[str, Any], lookup: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    explicit = list_field(term, "path_names")
    if explicit:
        return explicit
    result = []
    for uid in term.get("parent_ids", []) or []:
        parent = lookup.get(uid)
        if parent and parent.get("name"):
            result.append(parent["name"])
    return unique_texts(result)


def build_views(
    term: Mapping[str, Any],
    lookup: Mapping[str, Mapping[str, Any]],
    max_characters: int = 6000,
) -> Dict[str, str]:
    name = clean_text(term.get("name"))
    if not name:
        raise ValueError(f"empty name for {term.get('term_uid')}")
    sections = [name]
    code = clean_text(term.get("code"))
    if code:
        sections.append(f"Code: {code}")
    synonyms = unique_texts(
        list_field(term, "synonyms") + list_field(term, "index_terms")
    )
    if synonyms:
        sections.append("Synonyms: " + "; ".join(synonyms))
    description = clean_text(term.get("description"))
    if description and description.casefold() != name.casefold():
        sections.append("Description: " + description)
    definitions = [
        value for value in list_field(term, "definitions")
        if value.casefold() not in {name.casefold(), description.casefold()}
    ]
    if definitions:
        sections.append("Definitions: " + " ".join(definitions))
    parents = resolve_parent_names(term, lookup)
    if parents:
        sections.append("Parent concepts: " + "; ".join(parents))
    context = "\n".join(sections)
    return {
        "name_text": name,
        "context_text": context[:max_characters],
    }

