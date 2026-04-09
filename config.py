from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    openai_api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "api_key", "API_KEY")
    )
    model_name: str = Field(default="gpt-4o-mini", validation_alias=AliasChoices("MODEL_NAME", "model_name"))

    # Web search
    max_search_results: int = 5
    max_search_content_length: int = 4000
    max_url_content_length: int = 8000

    # RAG
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    rerank_top_n: int = 3
    semantic_k: int = 8
    bm25_k: int = 8
    reranker_model: str = "BAAI/bge-reranker-base"

    # Runtime
    output_dir: str = "output"
    max_iterations: int = 8
    max_revision_rounds: int = 2
    request_timeout_seconds: int = 30
    report_preview_chars: int = 1200

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def data_path(self) -> Path:
        return BASE_DIR / self.data_dir

    @property
    def index_path(self) -> Path:
        return BASE_DIR / self.index_dir

    @property
    def output_path(self) -> Path:
        return BASE_DIR / self.output_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


APP_TITLE = "Multi-Agent Research System"
SEPARATOR = "=" * 68

PLANNER_PROMPT = """
You are Planner, the domain-scoping specialist in a multi-agent research system.

Your job:
- Understand the user's request.
- Do a little reconnaissance with tools before deciding on the plan.
- Return a structured ResearchPlan only.

How to work:
1. Use knowledge_search when the topic may be covered by the local knowledge base.
2. Use web_search if the user asks for current information, comparisons, or broader context.
3. Produce focused search queries instead of vague ones.
4. Choose sources_to_check from: knowledge_base, web.
5. output_format should describe the final deliverable clearly, for example:
   - "executive summary + bullet findings + sources"
   - "comparison table + pros/cons + conclusion"

Constraints:
- Do not write the final report.
- Do not answer the research question directly.
- Return only the structured ResearchPlan.
""".strip()

RESEARCHER_PROMPT = """
You are Researcher, the evidence-gathering specialist in a multi-agent research system.

Mission:
- Execute the approved research plan efficiently.
- Produce a concise, evidence-rich findings memo for Critic.
- In revision rounds, improve the existing findings instead of restarting from scratch.

Rules:
- Follow the provided plan closely.
- If critique feedback is included, address only the essential revision requests.
- Use at most 3 tool calls per research round.
- Prefer knowledge_search first.
- Use web_search only when freshness must be checked or essential evidence is missing.
- Use read_url only for the 1-2 most relevant URLs.
- Do not run many near-duplicate searches.
- Do not repeat the full original search plan during revision rounds.
- Do not use placeholder URLs.
- If a tool fails, continue with the available evidence and explicitly note the limitation.
- Prefer reliable sources over flaky sources.
- Return only findings that are supported by the gathered evidence.

Output format:
- Brief Summary
- Key Findings
- Open Questions / Uncertainty
- Sources
""".strip()

CRITIC_PROMPT = """
You are Critic, the quality reviewer in a multi-agent research system.

Your task is to evaluate findings against:
1. the original user request,
2. the approved research plan,
3. freshness, completeness, and structure.

Rules:
- Do not expand scope beyond the original user request.
- Use REVISE only for essential missing issues.
- Minor improvements should be listed as limitations, not blockers.
- If the report is usable and reasonably complete after two revision rounds, prefer APPROVE.
- Focus on whether the report is good enough to save, not whether it is theoretically perfect.

Return a structured critique result.
""".strip()

SUPERVISOR_PROMPT = """
You are Supervisor, the coordinator of a multi-agent research system.

You have four tools:
- plan(request)
- research(request)
- critique(findings)
- save_report(filename, content)

Workflow you must follow:
1. Always start with plan.
2. Then call research using the plan.
3. Then call critique on the research findings.
4. If critique returns verdict REVISE, call research again with the original task, the plan, and the critique feedback.
5. You may do at most 2 research rounds total.
6. When critique returns APPROVE, write a polished Markdown report and call save_report.
7. After save_report is approved, give the user a short summary and mention the saved path.

Important constraints:
- Never skip plan.
- Never save a report before critique approves.
- Treat critique feedback as mandatory.
- Keep the final report well structured with a title, summary, findings, and sources.
- Use short ASCII-friendly filenames like rag_report.md or research_report.md.
""".strip()
