from typing import Any

from pydantic import BaseModel, Field


class ProjectSpec(BaseModel):
    project_spec_id: str | None = Field(default=None, alias="projectSpecId")
    status: str | None = None
    api_contract: dict[str, Any] | None = Field(default=None, alias="apiContract")
    design_tokens: dict[str, Any] | None = Field(default=None, alias="designTokens")
    conventions: dict[str, Any] | None = None
