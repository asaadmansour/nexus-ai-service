import logging

from fastapi import APIRouter, HTTPException

from app.agents.freelancer_matching import (
    MatchFreelancersRequest,
    match_freelancers,
)

router = APIRouter(prefix="/agents", tags=["Matching Agent"])

logger = logging.getLogger(__name__)


@router.post("/match-freelancers")
def match_freelancers_route(request: MatchFreelancersRequest):
    """Rank the freelancer candidates the backend provides for a given role."""
    try:
        return match_freelancers(request)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Freelancer matching failed.")
        raise HTTPException(
            status_code=503,
            detail="Freelancer matching is temporarily unavailable.",
        ) from exc
