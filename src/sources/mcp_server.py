from __future__ import annotations

import asyncio
import logging
import os
import re
from urllib.parse import unquote
import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

host = os.getenv("LAWCHAT_SOURCE_MCP_BIND_HOST", "127.0.0.1")
port = int(os.getenv("LAWCHAT_SOURCE_MCP_PORT", "8010"))

mcp = FastMCP(
    "LawChat Google Search",
    host=host,
    port=port,
)

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


async def _resolve_link(client: httpx.AsyncClient, item: dict) -> str | None:
    link = item.get("link") or ""

    # Check if about_page_link already contains the direct destination URL
    about_link = item.get("about_page_link") or ""
    m = re.search(r"q=About\+(https?://(?!www\.google\.com)[^\s&]+)", about_link)
    if m:
        return m.group(1)

    about_serp = item.get("about_page_serpapi_link") or ""
    m2 = re.search(r"q=About\+(https?%3A%2F%2F(?!www\.google\.com)[^\s&]+)", about_serp)
    if m2:
        return unquote(m2.group(1))

    # If it is already a direct non-Google URL, return it
    if (link.startswith("http://") or link.startswith("https://")) and "google.com/goto?" not in link:
        return link

    # Otherwise, resolve Google redirect link (/goto?url=... or https://www.google.com/goto?...)
    if link.startswith("/goto?"):
        target = f"https://www.google.com{link}"
    elif "google.com/goto?" in link:
        target = link
    else:
        return link or None

    try:
        resp = await client.get(target, follow_redirects=False, timeout=5.0)
        redirect_url = resp.headers.get("location")
        if redirect_url and redirect_url.startswith("http"):
            return redirect_url
    except Exception:
        pass

    return link or None


@mcp.tool()
async def google_search(query: str) -> dict:
    """Perform a Google search via SerpApi to find official legal source documents."""
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        logger.error("SERPAPI_API_KEY is not set.")
        return {"results": [], "error": "SERPAPI_API_KEY is not configured"}

    if not query or not query.strip():
        return {"results": []}

    timeout_seconds = float(os.getenv("LAWCHAT_SOURCE_MCP_TIMEOUT_SECONDS", "20"))

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": BROWSER_UA},
            timeout=timeout_seconds,
        ) as client:
            response = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": api_key,
                    "hl": "vi",
                    "gl": "vn",
                    "num": 10,
                },
            )
            response.raise_for_status()
            payload = response.json()

            organic = payload.get("organic_results", [])
            resolved_links = await asyncio.gather(
                *[_resolve_link(client, item) for item in organic],
                return_exceptions=True,
            )

            results = []
            for item, resolved in zip(organic, resolved_links):
                final_url = resolved if isinstance(resolved, str) and resolved else item.get("link")
                if not final_url:
                    continue
                results.append(
                    {
                        "url": final_url,
                        "title": item.get("title") or "",
                        "snippet": item.get("snippet") or "",
                    }
                )

            return {"results": results}
    except Exception as exc:
        logger.exception("Error querying SerpApi: %s", exc)
        return {"results": [], "error": str(exc)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting LawChat Source MCP server on %s:%s", host, port)
    mcp.run(transport="streamable-http")
