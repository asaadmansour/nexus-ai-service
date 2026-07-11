from fastapi import FastAPI
from app.routers.evaluation_agent import router as evaluation_agent_router
from app.routers.generate_task import router as generate_task_router
from app.routers.matching_agent import router as matching_agent_router
from app.routers.validate_brief import router as validate_brief_router
from app.routers.extract_cv import router as extract_cv_router

app = FastAPI()
app.include_router(evaluation_agent_router)
app.include_router(generate_task_router)
app.include_router(matching_agent_router)
app.include_router(validate_brief_router)
app.include_router(extract_cv_router)


@app.get("/health")
def health():
    return {
    "status": "ok",
    "service": "nexus-ai-service"
    }
