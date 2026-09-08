from pydantic import BaseModel, Field


class AudioSegment(BaseModel):
    scene_number: int
    text: str
    file: str
    duration_seconds: float = Field(ge=0)
    voice_profile: str
    rate: int = Field(ge=80, le=300)
    volume: float = Field(ge=0.0, le=2.0)


class NarrationManifest(BaseModel):
    provider: str
    voice_profile: str
    segments: list[AudioSegment] = Field(default_factory=list)
    total_duration_seconds: float = Field(ge=0)
