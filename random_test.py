from llm_graph.ChatModels.chatGroq import ChatGroq
from llm_graph.messages.ai import AIMessage
from llm_graph.messages.base import BaseMessage
from llm_graph.messages.human import HumanMessage

model = ChatGroq()

msg = BaseMessage(content="Hi",role="user")
human_msg = HumanMessage(content="hi")
ai_msg = AIMessage(content="You are a ai assistant named groq")
print(human_msg)
model.invoke(human_msg)

