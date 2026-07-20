import logging

from fastapi import APIRouter, HTTPException

from app.agents.project_quote_estimation import (
    ProjectQuoteEstimationError,
    ProjectQuoteRequest,
    estimate_project_quote,
)

router = APIRouter(prefix="/agents", tags=["Project Quote Estimation"])

logger = logging.getLogger(__name__)


@router.post("/estimate-project-quote")
def estimate_project_quote_route(request: ProjectQuoteRequest):
    """Estimate the customer-facing final project price after brief confirmation."""
    try:
        return estimate_project_quote(request)

    except ProjectQuoteEstimationError as exc:
        logger.exception("Project quote estimation failed: %s", str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except ValueError as exc:
        logger.warning("Invalid project quote input: %s", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Unexpected project quote estimation error")
        raise HTTPException(
            status_code=503,
            detail="Project quote estimation is temporarily unavailable.",
        ) from exc
