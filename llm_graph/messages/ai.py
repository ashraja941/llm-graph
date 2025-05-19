from llm_graph.messages.base import BaseMessage

class AIMessage(BaseMessage):
    role : str = "AI"
