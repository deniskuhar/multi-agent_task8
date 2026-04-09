from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import CRITIC_PROMPT, get_settings
from schemas import CritiqueResult
from tools import knowledge_search, read_url, web_search

settings = get_settings()
model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)

critic_web_search = tool(
    "web_search",
    description="Search the public web for relevant pages and return concise results with URLs.",
)(web_search)

critic_read_url = tool(
    "read_url",
    description="Read and extract the main textual content from a URL.",
)(read_url)

critic_knowledge_search = tool(
    "knowledge_search",
    description="Search the local knowledge base using hybrid retrieval and reranking.",
)(knowledge_search)

critic_agent = create_agent(
    model=model,
    tools=[critic_knowledge_search, critic_web_search, critic_read_url],
    system_prompt=CRITIC_PROMPT,
    response_format=CritiqueResult,
)