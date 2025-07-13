from typing import Literal
from pydantic import Field
from llm_graph.messages.base import BaseMessage

class AIMessage(BaseMessage):
    role : Literal["ai"] = Field("ai")
