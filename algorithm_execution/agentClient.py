import openai
from paper_search.token_counter import add_usage
class AgentClient:
    def __init__(self, llm="qwen-plus"):
        self.llm = llm
        self.client = openai.OpenAI(
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    def execute(self, query, system_prompt=None):
        response = self.client.chat.completions.create(
            model=self.llm,
            messages=[
                {"role":"system", 'content':system_prompt},{"role":"user","content":query} if system_prompt else  {"role":"user","content":query} ]
        )
        try:
            add_usage(getattr(response, "usage", None))
        except Exception:
            pass
        return response.choices[0].message.content
    

