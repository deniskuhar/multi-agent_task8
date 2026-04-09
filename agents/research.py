from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import RESEARCHER_PROMPT, get_settings
from tools import knowledge_search, read_url, web_search

settings = get_settings()
model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)

research_web_search = tool(
    "web_search",
    description="Search the public web for relevant pages and return concise results with URLs.",
)(web_search)

research_read_url = tool(
    "read_url",
    description="Read and extract the main textual content from a URL.",
)(read_url)

research_knowledge_search = tool(
    "knowledge_search",
    description="Search the local knowledge base using hybrid retrieval and reranking.",
)(knowledge_search)

researcher_agent = create_agent(
    model=model,
    tools=[research_knowledge_search, research_web_search, research_read_url],
    system_prompt=RESEARCHER_PROMPT,
)