from pydantic import BaseModel

class EduQueryInput(BaseModel):
    text: str

class EduQueryOutput(BaseModel):
    input: str
    output: str