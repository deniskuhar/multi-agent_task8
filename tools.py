from __future__ import annotations

import json
import re
import time
from typing import Any

import trafilatura
from ddgs import DDGS

from config import get_settings
from retriever import get_retriever

settings = get_settings()
OUTPUT_DIR = settings.output_path
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_RETRIEVER = None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _log_tool_start(name: str, **kwargs: Any) -> None:
    args_preview = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    print(f"\n🔧 Tool call: {name}({args_preview})")


def _log_tool_result(result: str) -> None:
    preview = result.replace("\n", " ")[:240]
    suffix = "..." if len(result) > 240 else ""
    print(f"📎 Result: {preview}{suffix}")


def web_search(query: str) -> str:
    _log_tool_start("web_search", query=query)
    results: list[dict[str, Any]] = []
    started = time.time()

    try:
        with DDGS(timeout=10) as ddgs:
            for item in ddgs.text(query, max_results=min(settings.max_search_results, 5)):
                if time.time() - started > 15:
                    break
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )
    except Exception as exc:
        result = f"Error searching web: {exc}"
        _log_tool_result(result)
        return result

    if not results:
        result = f"No web results found for query: {query}"
        _log_tool_result(result)
        return result

    result = _truncate(json.dumps(results, ensure_ascii=False, indent=2), settings.max_search_content_length)
    _log_tool_result(result)
    return result


def read_url(url: str) -> str:
    _log_tool_start("read_url", url=url)
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            result = f"Error: failed to download URL: {url}"
            _log_tool_result(result)
            return result
        extracted = trafilatura.extract(downloaded, include_links=True, include_formatting=False)
        if not extracted:
            result = f"Error: failed to extract readable content from URL: {url}"
            _log_tool_result(result)
            return result
        result = _truncate(extracted, settings.max_url_content_length)
        _log_tool_result(result)
        return result
    except Exception as exc:
        result = f"Error reading URL {url}: {exc}"
        _log_tool_result(result)
        return result


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename.strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "research_report"
    if not cleaned.lower().endswith(".md"):
        cleaned += ".md"
    return cleaned


def save_report(filename: str, content: str) -> str:
    _log_tool_start("save_report", filename=filename, content="...")
    try:
        safe_name = sanitize_filename(filename)
        path = OUTPUT_DIR / safe_name
        path.write_text(content, encoding="utf-8")
        result = f"Report saved to {path}"
        _log_tool_result(result)
        return result
    except Exception as exc:
        result = f"Error saving report: {exc}"
        _log_tool_result(result)
        return result


def write_report(filename: str, content: str) -> str:
    return save_report(filename=filename, content=content)


def knowledge_search(query: str) -> str:
    global _RETRIEVER
    _log_tool_start("knowledge_search", query=query)
    try:
        if _RETRIEVER is None:
            _RETRIEVER = get_retriever()
        docs = _RETRIEVER.hybrid_search(query)
    except Exception as exc:
        result = f"Error searching knowledge base: {exc}"
        _log_tool_result(result)
        return result

    if not docs:
        result = f"No local knowledge base results found for query: {query}"
        _log_tool_result(result)
        return result

    docs = docs[:5]

    lines = [f"Found {len(docs)} knowledge base results for query: {query}"]
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = _truncate(snippet, 500)
        lines.append(f"{idx}. [{source}{page_label}] {snippet}")
    result = "\n".join(lines)
    _log_tool_result(result)
    return result