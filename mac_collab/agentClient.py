from typing import *
import os
import json
import openai
from paper_search.token_counter import add_usage
from openai.types.chat.chat_completion import Choice
    
    

MODEL: str = "kimi-k2-0905-preview"
API_KEY: str = ""
BASE_URL: str = "https://api.moonshot.cn/v1"
TEMPERATURE: float = 0.3
class AgentClient:
    """Wrapper around the OpenAI API for GPT calls."""

    def __init__(self, api_key: str = API_KEY, model: str = MODEL, temperature: float = TEMPERATURE):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=self.api_key, base_url=BASE_URL)
       



    def chat_completion(self, messages: List[Dict[str, Any]],llm_model: str = None, temperature: float = None, max_tokens: int | None = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        model = llm_model if llm_model is not None else MODEL

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temp,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        # Add token usage to global counter
        try:
            add_usage(getattr(response, "usage", None))
        except Exception:
            pass
        message = response.choices[0].message.content
        return message

    def chat(self, messages) -> Choice:
        completion = self.client.chat.completions.create(
            model="kimi-k2-0711-preview",
            messages=messages,
            temperature=0.6,
            tools=[
                {
                    "type": "builtin_function",  
                    "function": {
                        "name": "$web_search",
                    },
                }
            ]
        )
        return completion.choices[0]

    def chat_completion_tool(self, messages: List[Dict[str, Any]], temperature: float = None) -> str:
        finish_reason = None
        while finish_reason is None or finish_reason == "tool_calls":
            choice = self.chat(messages)
            finish_reason = choice.finish_reason
            if finish_reason == "tool_calls":  
                messages.append(choice.message)  
                for tool_call in choice.message.tool_calls:  
                    tool_call_name = tool_call.function.name
                    tool_call_arguments = json.loads(tool_call.function.arguments)  
                    if tool_call_name == "$web_search":
                        tool_result = tool_call_arguments
                    else:
                        tool_result = f"Error: unable to find tool by name '{tool_call_name}'"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call_name,
                        "content": json.dumps(tool_result),  
                    })
    

        message = choice.message.content
        return message
