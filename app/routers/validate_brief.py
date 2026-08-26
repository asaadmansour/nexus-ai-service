import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.requirements.graph import requirements_graph

router = APIRouter(prefix="/agents", tags=["Requirements Agent"])
logger = logging.getLogger(__name__)


class ValidateBriefRequest(BaseModel):
    project_id: str = Field(alias="projectId")
    brief_id: str = Field(alias="briefId")
    latest_message: str | None = Field(default=None, alias="latestMessage")
    brief_text: str | None = Field(default=None, alias="briefText")
    current_brief: dict[str, Any] = Field(default_factory=dict, alias="currentBrief")
    recent_messages: list[dict[str, Any]] = Field(default_factory=list, alias="recentMessages")


@router.post("/validate-brief")
def validate_brief(request: ValidateBriefRequest):
    latest_message = request.latest_message or request.brief_text
    if not latest_message:
        raise HTTPException(
            status_code=400,
            detail="latestMessage is required.",
        )

    initial_state = {
        "projectId": request.project_id,
        "briefId": request.brief_id,
        "latestMessage": latest_message,
        "currentBrief": request.current_brief,
        "recentMessages": request.recent_messages,
    }
    try:
        final_state = requirements_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": request.brief_id}},
        )
    except Exception as exc:
        logger.exception("Requirements graph invocation failed.")
        raise HTTPException(
            status_code=503,
            detail="Requirements validation is temporarily unavailable.",
        ) from exc

    return {
        "isComplete": final_state.get("isComplete", False),
        "completionPercentage": final_state.get("completionPercentage", 0),
        "nextQuestion": final_state.get("nextQuestion"),
        "assistantReply": final_state.get("assistantReply"),
        "nextQuestionField": final_state.get("nextQuestionField"),
        "extractedFields": final_state.get("mergedBrief") or final_state.get("extractedFields", {}),
        "missingFields": final_state.get("missingFields", []),
        "fastPathUsed": final_state.get("fastPathUsed", False),
        "fastPathReason": final_state.get("fastPathReason"),
        "extractionSource": final_state.get("extractionSource", "llm"),
        "messageIntent": final_state.get("messageIntent"),
        "replyMode": final_state.get("replyMode"),
    }
