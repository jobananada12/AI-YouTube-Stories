from pydantic import BaseModel, Field


class SceneSpec(BaseModel):
    number: int
    title: str
    purpose: str
    narration: str
    estimated_duration_seconds: int = Field(ge=1)
    characters: list[str] = Field(default_factory=list)
    location: str
    time_of_day: str
    action: str
    visual_prompt: str
    mood: str
    continuity_notes: str = ""
    transition: str = "cut"


class ScenePlan(BaseModel):
    target_duration_seconds: int = Field(ge=1)
    total_duration_seconds: int = Field(ge=1)
    scenes: list[SceneSpec] = Field(default_factory=list)
