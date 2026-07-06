from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/agents", tags=["Requirements Agent"])


class ValidateBriefRequest(BaseModel):
    project_id: str = Field(alias="projectId")
    brief_id: str = Field(alias="briefId")
    latest_message: str = Field(alias="latestMessage")


@router.post("/validate-brief")
def validate_brief(request: ValidateBriefRequest):
    return {
        "isComplete": False,
        "nextQuestion": "What products will your store sell?",
        "extractedFields": {
            "project_type": "web_app",
            "domain": "e-commerce"
        },
        "missingFields": ["target_users", "payment_method", "delivery_scope"]
    }
