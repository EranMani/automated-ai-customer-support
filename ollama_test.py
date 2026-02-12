from pydantic import BaseModel

from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


class CityLocation(BaseModel):
    city: str
    country: str


ollama_model = OpenAIChatModel(
    model_name='llama3.2:latest',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),  
)
agent = Agent(ollama_model, output_type=PromptedOutput(CityLocation))
print(agent)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> RunUsage(input_tokens=57, output_tokens=8, requests=1)