from models.script_characterization import ScriptCharacterization

class LLMResponse:
    
    def __init__(self, content):
        self.content = content

class ScriptCharacterizerLLMResponse(LLMResponse):
    
    def __init__(self, content: ScriptCharacterization):
        super().__init__(content)
    
    
    
    
    