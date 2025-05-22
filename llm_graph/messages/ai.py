from typing import Literal
from llm_graph.messages.base import BaseMessage

class AIMessage(BaseMessage):
    role : Literal["ai"] = Field("ai")
