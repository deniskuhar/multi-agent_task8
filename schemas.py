from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    goal: str = Field(description="What we are trying to answer")
    search_queries: list[str] = Field(description="Specific queries to execute")
    sources_to_check: list[str] = Field(description="Choose from: knowledge_base, web")
    output_format: str = Field(description="What the final report should look like")


class CritiqueResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(description="Is the data up-to-date and based on recent sources?")
    is_complete: bool = Field(description="Does the research fully cover the user's original request?")
    is_well_structured: bool = Field(description="Are findings logically organized and ready for a report?")
    strengths: list[str] = Field(description="What is good about the research")
    gaps: list[str] = Field(description="What is missing, outdated, or poorly structured")
    revision_requests: list[str] = Field(description="Specific things to fix if verdict is REVISE")
