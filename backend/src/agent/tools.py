"""Research tools available to the agent loop. All free, no API keys.

Each Tool carries:
  spec     — neutral schema handed to the model (name, description, JSON-schema parameters)
  run      — executes the call and returns a ToolResult (text for the model + sources for the UI)
  running  — one-line present-tense status shown while the call is in flight
  done     — one-line past-tense label kept in the saved step list

Tool output is untrusted web content. The agent prompt tells the model to treat it
as data, and fetch_page refuses private / loopback addresses.
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from src.logger import get_logger

_log = get_logger(__name__)

_HTTP_TIMEOUT = 12.0
_PAGE_CHAR_LIMIT = 6000
# Sites like Wikipedia reject generic or spoofed agents; a descriptive one with contact info is accepted.
_USER_AGENT = f"LocalLLM-Research/1.0 (personal research assistant; +{os.getenv('RESEARCH_CONTACT', 'https://github.com/maxmoundas/local-llm')})"


@dataclass
class Source:
    title: str
    url: str


@dataclass
class ToolResult:
    text: str
    sources: list[Source] = field(default_factory=list)
    summary: str = ""  # short outcome for the step list, e.g. "6 results"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[dict], ToolResult]
    running: Callable[[dict], str]
    done: Callable[[dict], str]

    @property
    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _quote(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return f"“{text[:limit]}{'…' if len(text) > limit else ''}”"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _domain(url: str) -> str:
    host = urlparse(url).netloc
    return host.removeprefix("www.") or url


# ── web_search ────────────────────────────────────────────────────────────

def _ddgs_search(run: Callable[[], list | None]) -> tuple[list[dict], ToolResult | None]:
    """Run a DDGS query, turning its "no results" / rate-limit exceptions into advice for the planner."""
    from ddgs.exceptions import DDGSException, RatelimitException
    try:
        return run() or [], None
    except RatelimitException:
        return [], ToolResult(text="Search is rate-limited right now. Use the results you have, or read a page.",
                              summary="rate-limited")
    except DDGSException as exc:
        if "no results" in str(exc).lower():
            return [], ToolResult(text="No results. Try a shorter query with fewer keywords and no site: filter.",
                                  summary="0 results")
        raise


def _web_search(args: dict) -> ToolResult:
    from ddgs import DDGS
    query = str(args.get("query", "")).strip()
    if not query:
        return ToolResult(text="Error: empty query", summary="no query")
    results, problem = _ddgs_search(lambda: DDGS().text(query, max_results=6))
    if problem:
        return problem
    lines, sources = [], []
    for i, r in enumerate(results, 1):
        url = r.get("href", "")
        lines.append(f"[{i}] {r.get('title', '')}\n    {url}\n    {r.get('body', '')}")
        sources.append(Source(title=r.get("title", url), url=url))
    text = "\n".join(lines) or "No results."
    return ToolResult(text=text, sources=sources, summary=_count(len(results), "result"))


# ── news_search ───────────────────────────────────────────────────────────

_TIMELIMITS = {"day": "d", "week": "w", "month": "m"}


def _news_search(args: dict) -> ToolResult:
    from ddgs import DDGS
    query = str(args.get("query", "")).strip()
    if not query:
        return ToolResult(text="Error: empty query", summary="no query")
    timelimit = _TIMELIMITS.get(str(args.get("recency", "week")), "w")
    results, problem = _ddgs_search(lambda: DDGS().news(query, timelimit=timelimit, max_results=8))
    if problem:
        return problem
    lines, sources = [], []
    for i, r in enumerate(results, 1):
        url = r.get("url", "")
        date = str(r.get("date", ""))[:10]
        lines.append(f"[{i}] {r.get('title', '')} — {r.get('source', '')} ({date})\n    {url}\n    {r.get('body', '')}")
        sources.append(Source(title=r.get("title", url), url=url))
    text = "\n".join(lines) or "No recent news found."
    return ToolResult(text=text, sources=sources, summary=_count(len(results), "article"))


# ── fetch_page ────────────────────────────────────────────────────────────

def _is_public_host(host: str) -> bool:
    """Reject hosts that resolve to loopback, private, link-local, or reserved addresses."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) using lxml, dropping scripts, styles, and page chrome."""
    import lxml.html
    doc = lxml.html.fromstring(html)
    for bad in doc.xpath("//script|//style|//noscript|//nav|//footer|//header|//aside|//form|//svg"):
        bad.drop_tree()
    title = (doc.findtext(".//title") or "").strip()
    main = doc.xpath("//article") or doc.xpath("//main") or [doc]
    text = main[0].text_content()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return title, text


def _fetch_page(args: dict) -> ToolResult:
    url = str(args.get("url", "")).strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return ToolResult(text=f"Error: not a web URL: {url}", summary="invalid URL")
    if not _is_public_host(parsed.hostname):
        return ToolResult(text=f"Error: refusing to fetch non-public host {parsed.hostname}", summary="blocked host")

    with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True, headers={"User-Agent": _USER_AGENT}) as client:
        resp = client.get(url)
    final_host = resp.url.host
    if final_host and final_host != parsed.hostname and not _is_public_host(final_host):
        return ToolResult(text=f"Error: redirected to non-public host {final_host}", summary="blocked host")
    if resp.status_code >= 400:
        return ToolResult(text=f"Error: HTTP {resp.status_code} fetching {url}", summary=f"HTTP {resp.status_code}")

    content_type = resp.headers.get("content-type", "")
    if "html" in content_type:
        title, text = _html_to_text(resp.text)
    elif content_type.startswith("text/") or "json" in content_type:
        title, text = "", resp.text
    else:
        return ToolResult(text=f"Error: unsupported content type {content_type}", summary="unsupported")

    truncated = len(text) > _PAGE_CHAR_LIMIT
    text = text[:_PAGE_CHAR_LIMIT] + ("\n[…truncated]" if truncated else "")
    return ToolResult(
        text=f"Title: {title}\nURL: {resp.url}\n\n{text}",
        sources=[Source(title=title or _domain(url), url=str(resp.url))],
        summary=f"{len(text):,} chars",
    )


# ── get_weather ───────────────────────────────────────────────────────────

_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms", 96: "thunderstorms with hail",
    99: "severe thunderstorms with hail",
}


def _get_weather(args: dict) -> ToolResult:
    location = str(args.get("location", "")).strip()
    if not location:
        return ToolResult(text="Error: no location given", summary="no location")
    # Open-Meteo geocoding matches city names only, so drop any ", State" / ", Country" suffix.
    name = location.split(",")[0].strip()
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        geo = client.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": name, "count": 5, "language": "en"}).json()
        places = geo.get("results") or []
        if not places:
            return ToolResult(text=f"Could not find a place called {location}.", summary="not found")
        qualifier = location[len(name):].strip(" ,").lower()
        place = next((p for p in places if qualifier and qualifier in
                      f"{p.get('admin1', '')} {p.get('country', '')} {p.get('country_code', '')}".lower()), places[0])
        imperial = place.get("country_code") == "US"
        forecast = client.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 3, "timezone": "auto",
            "temperature_unit": "fahrenheit" if imperial else "celsius",
            "wind_speed_unit": "mph" if imperial else "kmh",
        }).json()

    t_unit, w_unit = ("°F", "mph") if imperial else ("°C", "km/h")
    cur = forecast["current"]
    label = ", ".join(x for x in (place.get("name"), place.get("admin1"), place.get("country")) if x)
    lines = [
        f"Weather for {label} (local time {cur['time']}):",
        f"Now: {cur['temperature_2m']}{t_unit} (feels like {cur['apparent_temperature']}{t_unit}), "
        f"{_WMO.get(cur['weather_code'], 'unknown')}, humidity {cur['relative_humidity_2m']}%, "
        f"wind {cur['wind_speed_10m']} {w_unit}, precipitation {cur['precipitation']} mm",
    ]
    d = forecast["daily"]
    for i, day in enumerate(d["time"]):
        lines.append(f"{day}: {_WMO.get(d['weather_code'][i], 'unknown')}, high {d['temperature_2m_max'][i]}{t_unit}, "
                     f"low {d['temperature_2m_min'][i]}{t_unit}, chance of precipitation {d['precipitation_probability_max'][i]}%")
    return ToolResult(
        text="\n".join(lines),
        sources=[Source(title=f"Open-Meteo forecast — {label}", url="https://open-meteo.com/")],
        summary=f"{cur['temperature_2m']}{t_unit}, {_WMO.get(cur['weather_code'], '')}",
    )


# ── Registry ──────────────────────────────────────────────────────────────

def default_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="Search the web. Returns titles, URLs, and snippets. Use specific queries.",
            parameters={"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"]},
            run=_web_search,
            running=lambda a: f"Searching the web for {_quote(a.get('query', ''))}",
            done=lambda a: f"Searched the web for {_quote(a.get('query', ''))}",
        ),
        Tool(
            name="news_search",
            description="Search recent news articles. Returns headlines, outlets, dates, URLs, and summaries.",
            parameters={"type": "object", "properties": {
                "query": {"type": "string", "description": "News search query"},
                "recency": {"type": "string", "enum": ["day", "week", "month"], "description": "How recent (default week)"},
            }, "required": ["query"]},
            run=_news_search,
            running=lambda a: f"Searching news for {_quote(a.get('query', ''))}",
            done=lambda a: f"Searched news for {_quote(a.get('query', ''))}",
        ),
        Tool(
            name="fetch_page",
            description="Read the main text of a web page. Use on promising URLs from search results.",
            parameters={"type": "object", "properties": {"url": {"type": "string", "description": "Full http(s) URL"}},
                        "required": ["url"]},
            run=_fetch_page,
            running=lambda a: f"Reading {_domain(str(a.get('url', '')))}",
            done=lambda a: f"Read {_domain(str(a.get('url', '')))}",
        ),
        Tool(
            name="get_weather",
            description="Current conditions and 3-day forecast for a city, e.g. 'Nashville' or 'Paris, France'.",
            parameters={"type": "object", "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"]},
            run=_get_weather,
            running=lambda a: f"Checking the weather in {a.get('location', '')}",
            done=lambda a: f"Checked the weather in {a.get('location', '')}",
        ),
    ]
