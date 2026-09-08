from pydantic import BaseModel, Field


class CharacterBibleEntry(BaseModel):
    name: str
    role: str
    age: str
    gender_presentation: str
    height: str
    build: str
    skin: str
    face: str
    eyes: str
    eyebrows: str
    nose: str
    lips: str
    hair: str
    facial_hair: str
    signature_clothing: str
    footwear: str
    accessories: str
    distinctive_features: str
    typical_expression: str
    color_palette: list[str] = Field(min_length=1, max_length=6)
    image_prompt_anchor: str


class CharacterBible(BaseModel):
    characters: list[CharacterBibleEntry] = Field(min_length=1)
