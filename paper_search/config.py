class llm_config:
    MODEL: str = "kimi-k2-0905-preview"
    API_KEY: str = ""
    BASE_URL: str = "https://api.moonshot.cn/v1"
    TEMPERATURE: float = 0.3
    tools = [
        {
            "type": "builtin_function",  
            "function": {
                "name": "$web_search",
            },
        },
    ]

