from pydantic import BaseModel, Field
    
class ScriptCharacterization(BaseModel):
    """
    Base class to model the output dto of the LLM
    """
    application: str = Field(description = "The name of the application executed by the commands.", max_length=25)
    
    
    
    