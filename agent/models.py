from pydantic import BaseModel, Field


class CompareDocuments(BaseModel):
    documents: list[str] = Field(
        description="Exactly two filenames from the available documents."
    )