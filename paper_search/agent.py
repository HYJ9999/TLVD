import json
from pathlib import Path

from openai import OpenAI, AsyncOpenAI
import openai
from .config import llm_config
from typing import Dict, Any, List
from openai.types.chat.chat_completion import Choice
from .token_counter import add_usage
class AgentClient:
    """Wrapper around the OpenAI API for GPT calls."""

    def __init__(self, api_key: str = llm_config.API_KEY, model: str = llm_config.MODEL, temperature: float = llm_config.TEMPERATURE):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=llm_config.BASE_URL)
        self.a_client = AsyncOpenAI(api_key=self.api_key, base_url=llm_config.BASE_URL)



    async def aclose(self):
        """Gracefully close underlying async/sync HTTP clients.
        Must be awaited before the event loop is closed to prevent httpx/anyio
        from raising 'RuntimeError: Event loop is closed'.
        """
        # Close async client first (httpx.AsyncClient under the hood)
        try:
            await self.a_client.aclose()  # AsyncOpenAI exposes aclose()
        except Exception:
            try:
                # Some versions may expose close() as an awaitable
                maybe = self.a_client.close()
                if hasattr(maybe, "__await__"):
                    await maybe
            except Exception:
                pass
        # Close sync client
        try:
            self.client.close()
        except Exception:
            pass

    def close(self):
        """Sync close for the sync client (safe to call multiple times)."""
        try:
            self.client.close()
        except Exception:
            pass

    async def a_chat_completion(self, messages: List[Dict[str, Any]], temperature: float = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        if "moonshot" in llm_config.MODEL or "kimi" in llm_config.MODEL:
            finish_reason = None


            while finish_reason is None or finish_reason == "tool_calls":
                response = await self.a_client.chat.completions.create(
                    model=llm_config.MODEL,
                    messages=messages,
                    temperature=temp,
                    tools=llm_config.tools
                )
                # record usage
                try:
                    add_usage(getattr(response, "usage", None))
                except Exception:
                    pass

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                if finish_reason == "tool_calls":  
                    messages.append(choice.message)  
                    for tool_call in choice.message.tool_calls:  
                        tool_call_name = tool_call.function.name
                        tool_call_arguments = json.loads(
                            tool_call.function.arguments)  
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

        elif "gpt" in llm_config.MODEL:
            openai.default_headers = {"x-foo": "true"}
            response = self.client.chat.completions.create(
                model=llm_config.MODEL,
                messages=messages,
                temperature=temp,
                tools=llm_config.tools
            )
            try:
                add_usage(getattr(response, "usage", None))
            except Exception:
                pass
            message = response.choices[0].message.content
        return message


    def chat_completion(self,messages):
        client = self.client
        completion = client.chat.completions.create(
            model=llm_config.MODEL,
            messages=messages,
            temperature=llm_config.TEMPERATURE,
        )
        try:
            add_usage(getattr(completion, "usage", None))
        except Exception:
            pass
        return completion.choices[0].message.content

    async def a_chat(messages) -> Choice:
        a_client = OpenAI(api_key=llm_config.API_KEY, base_url=llm_config.BASE_URL)
        completion = await a_client.chat.completions.create(
            model=llm_config.MODEL,
            messages=messages,
            temperature=llm_config.TEMPERATURE,
            tools=llm_config.tools
        )
        try:
            add_usage(getattr(completion, "usage", None))
        except Exception:
            pass
        return completion.choices[0]
