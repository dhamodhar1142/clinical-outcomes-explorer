from __future__ import annotations

import re
from typing import Any


FIELD_ALIASES = {
    "first_name": ["first name", "firstname", "given name"],
    "last_name": ["last name", "lastname", "family name", "surname"],
    "email": ["email", "e-mail"],
    "phone": ["phone", "mobile", "telephone"],
    "city": ["city", "town"],
    "location": ["location", "address", "state"],
    "linkedin_url": ["linkedin", "linkedin profile"],
    "website_url": ["website", "portfolio"],
    "sponsorship": ["sponsorship", "sponsor", "visa"],
    "work_authorization": ["work authorization", "authorized", "eligible to work"],
}


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def score_field_descriptor(descriptor: dict[str, str], canonical_name: str) -> float:
    haystacks = [
        normalize_text(descriptor.get("label", "")),
        normalize_text(descriptor.get("name", "")),
        normalize_text(descriptor.get("id", "")),
        normalize_text(descriptor.get("placeholder", "")),
        normalize_text(descriptor.get("nearby_text", "")),
    ]
    aliases = FIELD_ALIASES.get(canonical_name, [])
    best_score = 0.0
    for alias in aliases:
        alias_norm = normalize_text(alias)
        for index, haystack in enumerate(haystacks):
            if alias_norm and alias_norm in haystack:
                weight = [1.0, 0.85, 0.75, 0.6, 0.5][index]
                best_score = max(best_score, weight)
    return best_score


def map_fields(descriptors: list[dict[str, str]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for descriptor in descriptors:
        candidates = []
        for canonical in FIELD_ALIASES:
            score = score_field_descriptor(descriptor, canonical)
            if score > 0:
                candidates.append((canonical, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if not candidates:
            results.append({"field": descriptor, "mapped_to": None, "confidence": 0.0, "value": None})
            continue
        mapped_to, confidence = candidates[0]
        value = profile.get(mapped_to)
        if mapped_to == "work_authorization":
            value = "Yes" if profile.get("work_authorization", {}).get("authorized_us", True) else "No"
        elif mapped_to == "sponsorship":
            value = "Yes" if profile.get("work_authorization", {}).get("need_sponsorship_now", False) else "No"
        results.append({"field": descriptor, "mapped_to": mapped_to, "confidence": confidence, "value": value})
    return results
