from dataclasses import dataclass, field
from typing import List


@dataclass
class Character:
    name: str
    role: str
    description: str
    personality: str = ""


@dataclass
class Scene:
    number: int
    title: str
    narration: str
    visual_prompt: str
    duration_seconds: float = 0.0
    characters: List[str] = field(default_factory=list)


@dataclass
class Story:
    title: str
    genre: str
    logline: str
    script: str
    characters: List[Character] = field(default_factory=list)
    scenes: List[Scene] = field(default_factory=list)
