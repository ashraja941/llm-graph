from llm_graph.messages.base import BaseMessage
from typing import Literal
from pydantic import Field

class HumanMessage(BaseMessage):
    role : Literal["user"] = Field("user") 
