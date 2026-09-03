from models.script import Script
from pydantic import BaseModel
from models.script_characterization import ScriptCharacterization, ScriptTaxonomyCharacterization, ScriptStepsCharacterization

class LLMRequest:
        
    def __init__(self, prompt: str, system_message:str, struct: BaseModel | dict = None):
        """Parameters
        ----------
        prompt : str
            The main user prompt or query to send to the model.
        struct : type or dict, optional
            A structure definition (e.g., a Pydantic model, dataclass, or schema)
            that specifies the expected shape of the output. If provided, the
            method attempts to parse or coerce the model’s response into the
            given structure.
        """
        self.prompt = prompt
        self.struct = struct
        self.system_message = system_message    

class ScriptCharacterizerLLMRequest(LLMRequest):
    """
    Class to format the requests to characterize the job's application.
    """
    
    def __init__(self, script: Script, system_message:str):
        """
        Parameters
        ----------
        script: Script
            The script to analyse.
        max_len: int, optional
            Maximum length of the LLM response application name.
            Overrides the ScriptCharacterization default when provided.
        """
        prompt = f'SCRIPT:"\n{script.script_commands}"\nOUTPUT:\n'
        super().__init__(prompt, system_message, ScriptCharacterization)


class ScriptTaxonomyLLMRequest(LLMRequest):
    """
    Class to format the requests to characterize the job with the taxonomy structure.
    """

    def __init__(self, script: Script, system_message:str):
        """
        Parameters
        ----------
        script: Script
            The script to analyse.
        system_message: str
            The system prompt to use.
        """
        prompt = f'SCRIPT:"\n{script.script_commands}"\nOUTPUT:\n'
        super().__init__(prompt, system_message, ScriptTaxonomyCharacterization)


class ScriptStepsLLMRequest(LLMRequest):
    """
    Class to format the requests to characterize the job with the step-based structure.
    """

    def __init__(self, script: Script, system_message: str):
        """
        Parameters
        ----------
        script: Script
            The script to analyse.
        system_message: str
            The system prompt to use.
        """
        prompt = f'SCRIPT:"\n{script.script_commands}"\nOUTPUT:\n'
        super().__init__(prompt, system_message, ScriptStepsCharacterization)
        