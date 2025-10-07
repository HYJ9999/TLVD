import openai
from .config import llm_config
from paper_search.token_counter import add_usage
class AgentClient:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key = llm_config.api_key,
            base_url = llm_config.base_url
        )
    def execute(self, query, system_prompt=None, llm="deepseek-v3"):
        response = self.client.chat.completions.create(
            model=llm,
            messages=[
                {"role":"system", 'content':system_prompt},{"role":"user","content":query} if system_prompt else  {"role":"user","content":query} ]
        )
        try:
            add_usage(getattr(response, "usage", None))
        except Exception:
            pass
        return response.choices[0].message.content
    

