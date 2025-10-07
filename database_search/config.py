class llm_config:
    api_key : str = ""
    base_url : str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class neo4j_config:
    # Create driver
    uri = "neo4j://localhost:7687"  # Local default port, or "neo4j://..." for encrypted connection
    username = ""
    password = ""
    database = ""
