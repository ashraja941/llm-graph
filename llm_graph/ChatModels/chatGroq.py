import os
from typing import Union
from groq import Groq

from llm_graph.messages.ai import AIMessage
from llm_graph.messages.base import BaseMessage
from llm_graph.messages.human import HumanMessage

class ChatGroq():
    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing GROQ_API_KEY")

        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def invoke(self,message:Union[BaseMessage,AIMessage,HumanMessage]) -> str:
        chat_completion = self.client.chat.completions.create(
            messages = [
                message.to_dict(),
            ],
            model = self.model
        )
        print(chat_completion.choices[0].message.content)
        return str(chat_completion.choices[0].message.content)
