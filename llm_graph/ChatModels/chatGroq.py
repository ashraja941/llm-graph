import os
from groq import Groq

class ChatGroq():
    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing GROQ_API_KEY")

        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def invoke(self,message) -> str:
        chat_completion = self.client.chat.completions.create(
            messages = [
                {
                    "role" : "user",
                    "content" : message
                }
            ],
            model = self.model
        )
        print(chat_completion.choices[0].message.content)
        return str(chat_completion.choices[0].message.content)
