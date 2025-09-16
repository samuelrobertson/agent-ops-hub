from typing import List, Literal, Dict, Any
from pydantic import BaseModel

class State(BaseModel):
    question: str
    route: Literal["qa","skill"] | None = None
    retrieved: List[Dict[str, Any]] | None = None
    answer: str | None = None
    citations: List[Dict[str, Any]] | None = None
