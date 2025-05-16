import os
from openai import OpenAI

class ChatOpenAI():
    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing OPENAI_API_KEY")

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    def invoke(self,message):
        response = self.client.responses.create(
                model = self.model,
                input=message,
                )
        print("response : ",response)
        return response.output_text
