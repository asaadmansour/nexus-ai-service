import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.routers.evaluation_agent import router as evaluation_agent_router
from app.routers.generate_task import router as generate_task_router
from app.routers.matching_agent import router as matching_agent_router
from app.routers.validate_brief import router as validate_brief_router
from app.routers.extract_cv import router as extract_cv_router
from app.routers.generate_assessment import router as generate_assessment_router
from app.routers.grade_assessment import router as grade_assessment_router
from app.routers.generate_embedding import router as generate_embedding_router
from app.routers.generate_project_plan import router as generate_project_plan_router
from app.routers.evaluate_planning_submission import router as evaluate_planning_submission_router
from app.routers.generate_role_brief import router as generate_role_brief_router
from app.routers.estimate_project_quote import router as estimate_project_quote_router


app = FastAPI()
app.include_router(evaluation_agent_router)
app.include_router(generate_task_router)
app.include_router(matching_agent_router)
app.include_router(validate_brief_router)
app.include_router(extract_cv_router)
app.include_router(generate_assessment_router)
app.include_router(grade_assessment_router)
app.include_router(generate_embedding_router)
app.include_router(generate_project_plan_router)
app.include_router(evaluate_planning_submission_router)
app.include_router(generate_role_brief_router)
app.include_router(estimate_project_quote_router)


@app.get("/health")
def health():
    configured = is_ai_provider_configured()
    payload = {
        "status": "ok" if configured else "degraded",
        "service": "nexus-ai-service",
        "aiProviderConfigured": configured,
    }
    return payload if configured else JSONResponse(status_code=503, content=payload)


@app.get("/health/live")
def liveness():
    return {"status": "ok", "service": "nexus-ai-service"}


def is_ai_provider_configured() -> bool:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(api_key and api_key != "change-me")
