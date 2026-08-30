from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to retrieve for")
    top_k: int = Field(5, ge=1, le=20, description="Passages to return")


class Passage(BaseModel):
    number: int = Field(..., description="The number the answer cites this passage by")
    arxiv_id: str = Field(..., description="Identifier including its version suffix")
    section: str
    paragraph: int = Field(..., description="Index of the paragraph within the paper")
    text: str
    score: float


class SearchResponse(BaseModel):
    question: str
    passages: list[Passage]
