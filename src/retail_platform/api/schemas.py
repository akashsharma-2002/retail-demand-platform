from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    store_id: str = Field(pattern=r"^[A-Z]{2}_\d$")
    budget: float | None = Field(default=None, gt=0, le=10_000_000)
    demand_change_pct: float = Field(default=0.0, ge=-90, le=300)
    lead_time_change_days: int = Field(default=0, ge=-14, le=30)


class DecisionRequest(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=64)
    store_id: str | None = Field(default=None, pattern=r"^[A-Z]{2}_\d$")


class ResumeRequest(BaseModel):
    thread_id: str = Field(max_length=64)
    approve: bool
    note: str | None = Field(default=None, max_length=500)
