from backend.app.models.schemas import ApplyPlanRequest, JobParseRequest, Profile
from backend.app.services.apply_plan_service import ApplyPlanService
from backend.app.services.field_mapper import map_fields
from backend.app.services.jd_parser import JDParserService
from backend.app.services.orchestration_service import OrchestrationService


def build_profile() -> Profile:
    return Profile(
        full_name="Taylor Candidate",
        email="taylor@example.com",
        phone="5551234567",
        location="Chicago, IL",
        linkedin_url="https://linkedin.com/in/taylor",
        master_resume_text=(
            "Summary\nAnalytics leader.\n\n"
            "Technical Skills\nSQL, Python\n\n"
            "Education\nState University\n\n"
            "Professional Experience\nHealthCo\n- Improved reports.\n\n"
            "Projects\n- Claims dashboard."
        ),
        default_answers={"veteran_status": "Decline to answer", "disability_status": "Decline to answer"},
    )


def test_jd_parser_detects_portal_and_keywords():
    service = JDParserService()
    result = service.parse(
        JobParseRequest(
            source_url="https://company.greenhouse.io/jobs/123",
            raw_text="Senior Healthcare Data Analyst Responsibilities: build reporting, analyze claims, improve payment insights. Requirements: SQL, SAS, dashboards.",
        )
    )
    assert result.portal_type == "greenhouse"
    assert "sql" in result.keywords
    assert result.title


def test_apply_plan_defaults_to_review_first():
    parser = JDParserService()
    job = parser.parse(
        JobParseRequest(
            source_url="https://example.com/jobs/1",
            raw_text="Healthcare analyst role with SQL and reporting responsibilities.",
        )
    )
    plan = ApplyPlanService().build(ApplyPlanRequest(job=job, profile=build_profile()))
    assert plan.submit_mode == "review_first"
    assert any(step.action == "review" for step in plan.steps)


def test_field_mapper_prefers_labels():
    descriptors = [
        {"label": "First Name", "name": "fname", "id": "fname", "placeholder": "", "nearby_text": ""},
        {"label": "Work Authorization", "name": "auth", "id": "auth", "placeholder": "", "nearby_text": ""},
    ]
    mapped = map_fields(descriptors, build_profile().model_dump())
    assert mapped[0]["mapped_to"] == "first_name"
    assert mapped[1]["mapped_to"] == "work_authorization"


def test_orchestration_creates_artifacts():
    service = OrchestrationService()
    profile = build_profile()
    result = service.run(
        {
            "source_url": "https://jobs.lever.co/example/123",
            "raw_text": "Senior Healthcare Analyst Responsibilities: build claims reporting and SQL dashboards. Requirements: SQL, analytics, healthcare, stakeholder communication.",
            "profile": profile,
            "submit_mode": "review_first",
            "run_worker": False,
        }
    )
    assert result.worker_status == "review_ready"
    assert len(result.artifacts) == 5
