from openai import OpenAI # AsyncOpenAI
from dtos.llm_request import LLMRequest
from dtos.llm_response import LLMResponse 

class LLMClient:
    
    def __init__(self, endpoint:str, api_key:str, model:str, timeout:int = 600):
        # Create connection to the internal system
        self.client = OpenAI(
            base_url=endpoint,
            api_key = api_key,
            timeout=timeout,
        )
        self.model = model
    
    def query_model(self, request:LLMRequest, response_type: LLMResponse = LLMResponse) -> LLMResponse:
        """
        Send a request to the underlying language model and return its response.

        Parameters
        ----------
        request : LLMRequest
            The request to send to the model.
        response_type: LLMResponse
            The type of response, defaults to LLMResponse
            
        Returns
        -------
        LLMResponse
            The raw model response or a structured object if `struct` is supplied
            and parsing is successful.

        """
        messages = []
        if request.system_message:
            messages.append({"role":"system", "content":request.system_message})
        messages.append({"role": "user", "content": request.prompt})
        # Setup structured output
        if request.struct:
            # Call parse API
            completion = self.client.chat.completions.parse(
                response_format = request.struct, 
                model = self.model,
                messages = messages,
            )
            response = completion.choices[0].message.parsed
        else:
            # Call create API if structured output not needed
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            response = completion.choices[0].message.content
        return response_type(content = response)
