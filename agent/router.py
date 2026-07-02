from typing import Literal
from pydantic import BaseModel, Field


class Route(BaseModel):
    tool: Literal[
        "qa",
        "summary",
        "compare",
        "extract",
        "search_all"
    ]