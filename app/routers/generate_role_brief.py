import logging

from fastapi import APIRouter, HTTPException

from app.agents.role_brief_generation import (
    RoleBriefGenerationError,
    RoleBriefRequest,
    generate_role_brief,
)

router = APIRouter(prefix="/agents", tags=["Role Brief Generation"])

logger = logging.getLogger(__name__)


@router.post("/generate-role-brief")
def generate_role_brief_route(request: RoleBriefRequest):
    """Generate role-specific assignment requirements for planning freelancers."""
    try:
        return generate_role_brief(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RoleBriefGenerationError as exc:
        logger.exception("Role brief generation failed.")
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "30"},
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected role brief generation failure.")
        raise HTTPException(
            status_code=503,
            detail="Role brief generation is temporarily unavailable.",
        ) from exc
