from typing import Optional
from pydantic import BaseModel

class BaseMessage(BaseModel):
    content : str 
    role : str

    def to_dict(self):
        return self.__dict__
