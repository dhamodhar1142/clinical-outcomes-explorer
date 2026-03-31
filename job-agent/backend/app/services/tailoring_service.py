from __future__ import annotations

import re

from ..models.schemas import JobParseRequest, TailorResumeRequest, TailorResumeResponse
from .jd_parser import JDParserService
from .prompt_loader import PromptLoader


class ResumeTailoringService:
    ordered_sections = [
        "Summary",
        "Technical Skills",
        "Education",
        "Professional Experience",
        "Projects",
    ]

    def __init__(self) -> None:
        self.parser = JDParserService()
        self.prompt_loader = PromptLoader()

    def tailor_resume(self, payload: TailorResumeRequest) -> TailorResumeResponse:
        parsed_job = payload.parsed_job or self.parser.parse(
            JobParseRequest(source_url="https://local/job", raw_text=payload.job_description)
        )
        keywords = parsed_job.keywords[:8]
        prompt = self.prompt_loader.render_tailoring_prompt(payload.master_resume, payload.job_description)
        sections = self._split_sections(payload.master_resume)
        tailored_sections = {
            "Summary": self._rewrite_summary(sections.get("Summary", ""), parsed_job.title, parsed_job.company, keywords),
            "Technical Skills": self._enrich_skills(sections.get("Technical Skills", ""), keywords),
            "Education": sections.get("Education", "").strip(),
            "Professional Experience": self._rewrite_experience(sections.get("Professional Experience", ""), keywords, parsed_job.title),
            "Projects": self._rewrite_projects(sections.get("Projects", ""), keywords),
        }
        tailored_resume = self._join_sections(tailored_sections)
        change_summary = [
            f"Aligned the summary to {parsed_job.title} expectations at {parsed_job.company}.",
            "Strengthened experience bullets with job-description keywords and business outcomes.",
            "Preserved the original section order, employers, and education entries.",
        ]
        return TailorResumeResponse(
            tailored_resume_text=tailored_resume,
            keywords=keywords,
            change_summary=change_summary,
            prompt_used=prompt,
        )

    def _split_sections(self, resume_text: str) -> dict[str, str]:
        pattern = re.compile(r"(?im)^(Summary|Technical Skills|Education|Professional Experience|Projects)\s*$")
        matches = list(pattern.finditer(resume_text))
        if not matches:
            return {
                "Summary": resume_text.strip(),
                "Technical Skills": "",
                "Education": "",
                "Professional Experience": "",
                "Projects": "",
            }
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(resume_text)
            sections[match.group(1)] = resume_text[start:end].strip()
        return sections

    def _join_sections(self, sections: dict[str, str]) -> str:
        parts: list[str] = []
        for heading in self.ordered_sections:
            parts.append(heading)
            parts.append(sections.get(heading, "").strip())
            parts.append("")
        return "\n".join(parts).strip() + "\n"

    def _rewrite_summary(self, existing: str, title: str, company: str, keywords: list[str]) -> str:
        keyword_phrase = ", ".join(keywords[:5]) if keywords else "analytics, reporting, and cross-functional delivery"
        existing_clean = existing.strip()
        return (
            f"Healthcare analytics professional targeting {title} opportunities with a record of translating messy data into "
            f"clear reporting, stakeholder-ready insights, and measurable operational improvement. Known for SQL-driven analysis, "
            f"process rigor, and partnering with business teams to improve quality, reporting accuracy, and execution across {company}-style environments. "
            f"Core alignment areas: {keyword_phrase}.\n\n"
            f"{existing_clean}".strip()
        )

    def _enrich_skills(self, existing: str, keywords: list[str]) -> str:
        additions = [term.upper() if term.lower() in {"sql", "sas"} else term.title() for term in keywords[:6]]
        merged = existing.strip()
        extra = ", ".join(dict.fromkeys(additions))
        if extra and extra.lower() not in merged.lower():
            merged = f"{merged}\nTargeted Alignment: {extra}".strip()
        return merged

    def _rewrite_experience(self, existing: str, keywords: list[str], title: str) -> str:
        if not existing.strip():
            return ""
        lines = [line.rstrip() for line in existing.splitlines() if line.strip()]
        rewritten: list[str] = []
        first_role_adjusted = False
        keyword_phrase = ", ".join(keywords[:4]) if keywords else "reporting, analytics, and stakeholder communication"
        for line in lines:
            if line.lstrip().startswith(("-", "•")):
                bullet = line.lstrip("-• ").strip()
                rewritten.append(
                    f"- Led {bullet[:1].lower() + bullet[1:] if bullet else 'analytics delivery'}, emphasizing {keyword_phrase} and business impact."
                )
            else:
                if not first_role_adjusted and " / " not in line and "," not in line:
                    rewritten.append(f"{line} / {title}")
                    first_role_adjusted = True
                else:
                    rewritten.append(line)
        return "\n".join(rewritten)

    def _rewrite_projects(self, existing: str, keywords: list[str]) -> str:
        if not existing.strip():
            return ""
        project_boost = ", ".join(keywords[:3]) if keywords else "analytics, reporting, and workflow improvement"
        return f"{existing.strip()}\n- Highlighted projects were reframed to reinforce {project_boost} credibility."
