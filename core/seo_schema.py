from pydantic import BaseModel, Field


class YouTubeMetadata(BaseModel):
    title: str = Field(max_length=100)
    description: str
    keywords: list[str] = Field(min_length=5, max_length=30)
    tags: list[str] = Field(min_length=5, max_length=30)
    hashtags: list[str] = Field(min_length=3, max_length=15)
