from typing import List
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str


class ChatInput(BaseModel):
    model: str
    messages: List[Message]
