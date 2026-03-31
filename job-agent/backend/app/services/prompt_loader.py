from pathlib import Path


class PromptLoader:
    def __init__(self) -> None:
        self.prompt_dir = Path(__file__).resolve().parents[1] / "prompts"

    def load(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8")

    def render_tailoring_prompt(self, master_resume: str, job_description: str) -> str:
        template = self.load("tailoring_prompt.txt")
        return (
            f"{template}\n\n"
            f"MASTER RESUME:\n{master_resume.strip()}\n\n"
            f"JOB DESCRIPTION:\n{job_description.strip()}\n"
        )
