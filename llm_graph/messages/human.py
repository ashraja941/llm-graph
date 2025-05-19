from llm_graph.messages.base import BaseMessage
class HumanMessage(BaseMessage):
    role : str = "Human"
