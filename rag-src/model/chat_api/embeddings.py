from pydantic import BaseModel

class EmbeddingsRequest(BaseModel):
    input: list
    model: str

class EmbeddingsResponse(BaseModel):
    data: list

