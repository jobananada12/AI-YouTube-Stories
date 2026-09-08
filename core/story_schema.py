from pydantic import BaseModel, Field


class CharacterSpec(BaseModel):
    name: str
    role: str
    age: str
    appearance: str
    personality: str
    motivation: str


class StorySpec(BaseModel):
    title: str
    genre: str
    tone: str
    logline: str
    premise: str
    theme: str
    setting: str
    protagonist_goal: str
    central_conflict: str
    ending_type: str
    characters: list[CharacterSpec] = Field(min_length=2, max_length=10)
    outline: list[str] = Field(min_length=8, max_length=24)
