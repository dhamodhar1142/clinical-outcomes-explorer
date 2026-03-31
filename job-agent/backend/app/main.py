from fastapi import FastAPI

from .api import apply, jobs, tailor


app = FastAPI(
    title="AI Job Application Agent",
    version="0.1.0",
    description="Review-first backend for job parsing, tailoring, document rendering, and apply-plan orchestration.",
)

app.include_router(jobs.router)
app.include_router(tailor.router)
app.include_router(apply.router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
