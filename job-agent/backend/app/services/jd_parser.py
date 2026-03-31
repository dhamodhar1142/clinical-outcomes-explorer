from __future__ import annotations

import re
from urllib.parse import urlparse

from ..models.schemas import JobParseRequest, JobParseResponse, PortalType


class JDParserService:
    keyword_bank = [
        "healthcare",
        "analytics",
        "analyst",
        "sql",
        "sas",
        "python",
        "claims",
        "payment",
        "payments",
        "revenue cycle",
        "reporting",
        "dashboards",
        "etl",
        "data quality",
        "medicare",
        "medicaid",
        "utilization",
        "kpi",
        "stakeholder",
    ]

    def parse(self, payload: JobParseRequest) -> JobParseResponse:
        text = self._normalize(payload.raw_text)
        portal_type = self._detect_portal(payload.source_url)
        title = payload.title_hint or self._extract_title(text) or "Unknown Role"
        company = payload.company_hint or self._extract_company(text, payload.source_url)
        location = payload.location_hint or self._extract_location(text)
        responsibilities = self._extract_bullets(text, ("responsibilities", "what you will do", "duties"))
        requirements = self._extract_bullets(text, ("requirements", "qualifications", "what you bring", "must have"))
        keywords = self._extract_keywords(text)
        apply_url = payload.apply_links[0] if payload.apply_links else payload.source_url
        return JobParseResponse(
            title=title,
            company=company,
            location=location,
            responsibilities=responsibilities,
            requirements=requirements,
            keywords=keywords,
            apply_url=apply_url,
            portal_type=portal_type,
            raw_text_preview=text[:1200],
            source_url=payload.source_url,
        )

    def _normalize(self, raw_text: str) -> str:
        return re.sub(r"\s+", " ", raw_text).strip()

    def _detect_portal(self, url: str) -> PortalType:
        hostname = urlparse(url).netloc.lower()
        if "myworkdayjobs" in hostname or "workday" in hostname:
            return "workday"
        if "greenhouse" in hostname:
            return "greenhouse"
        if "lever" in hostname:
            return "lever"
        return "generic"

    def _extract_title(self, text: str) -> str | None:
        patterns = [
            r"(senior|lead|principal|staff)?\s*(healthcare|data|business|reporting)?\s*(analyst|scientist|manager|engineer)",
            r"(job title|title)\s*[:\-]\s*([A-Za-z0-9 /&,-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            if match.lastindex and match.lastindex > 1:
                return match.group(match.lastindex).strip().title()
            return match.group(0).strip().title()
        return None

    def _extract_company(self, text: str, url: str) -> str:
        company_match = re.search(r"(company|employer)\s*[:\-]\s*([A-Za-z0-9 .,&'-]+)", text, flags=re.IGNORECASE)
        if company_match:
            return company_match.group(2).strip()
        hostname = urlparse(url).netloc.lower().split(":")[0]
        parts = [part for part in hostname.split(".") if part not in {"www", "jobs", "careers", "boards"}]
        return parts[0].replace("-", " ").title() if parts else "Unknown Company"

    def _extract_location(self, text: str) -> str:
        location_match = re.search(
            r"(location|based in|office)\s*[:\-]?\s*([A-Za-z .]+,\s?[A-Z]{2}|remote|hybrid|onsite)",
            text,
            flags=re.IGNORECASE,
        )
        return location_match.group(2).strip().title() if location_match else "Unspecified"

    def _extract_bullets(self, text: str, headings: tuple[str, ...]) -> list[str]:
        collected: list[str] = []
        for heading in headings:
            match = re.search(rf"{heading}\s*[:\-]?\s*(.+?)(?:\b(requirements|qualifications|benefits|about us)\b|$)", text, re.IGNORECASE)
            if not match:
                continue
            sentence_blob = match.group(1)
            sentences = re.split(r"(?<=[.!?])\s+", sentence_blob)
            for sentence in sentences:
                cleaned = sentence.strip(" -•")
                if len(cleaned) > 20:
                    collected.append(cleaned)
            if collected:
                break
        return collected[:6]

    def _extract_keywords(self, text: str) -> list[str]:
        lowered = text.lower()
        found = [keyword for keyword in self.keyword_bank if keyword in lowered]
        words = re.findall(r"[a-zA-Z][a-zA-Z+/ -]{2,}", lowered)
        freq: dict[str, int] = {}
        for word in words:
            normalized = word.strip()
            if len(normalized) < 4:
                continue
            freq[normalized] = freq.get(normalized, 0) + 1
        ranked = [word for word, _count in sorted(freq.items(), key=lambda item: (-item[1], item[0]))]
        merged: list[str] = []
        for term in found + ranked:
            if term not in merged:
                merged.append(term)
            if len(merged) == 12:
                break
        return merged
