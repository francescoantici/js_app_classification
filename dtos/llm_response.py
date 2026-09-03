from models.script_characterization import ScriptCharacterization, ScriptTaxonomyCharacterization, ScriptStepsCharacterization

class LLMResponse:

    def __init__(self, content):
        self.content = content

class ScriptCharacterizerLLMResponse(LLMResponse):

    def __init__(self, content: ScriptCharacterization):
        super().__init__(content)

class ScriptTaxonomyLLMResponse(LLMResponse):

    def __init__(self, content: ScriptTaxonomyCharacterization):
        super().__init__(content)

class ScriptStepsLLMResponse(LLMResponse):

    def __init__(self, content: ScriptStepsCharacterization):
        super().__init__(content)
    
    
    
    
    