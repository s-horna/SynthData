"""LLM-backed mock MCP server (Phase 2).

Exposes whatever toolset it is given and answers calls with the cheap model:
  - arguments are validated with jsonschema; invalid ones get a realistic MCP error
  - a configurable fraction of valid calls get an injected error
    (timeout / not found / permission denied / rate limit)
  - seeded, deterministic mode with an on-disk response cache

Run as a stdio MCP server (this is how Data Designer launches it):
    python -m mock_server.server --toolset path/to/toolset.json [--error-rate 0.07] [--offline]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import anyio
import httpx
import mcp.types as types
from jsonschema import validators
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import api_key, load_config, resolve  # noqa: E402

TRANSIENT = {"timeout", "rate_limit"}

ERROR_TEMPLATES = {
    "timeout": "Error: upstream request timed out after {secs}s while executing '{tool}'. The operation may be retried.",
    "not_found": "Error 404: the requested resource was not found ({hint}).",
    "permission_denied": "Error 403: permission denied. The current credentials lack the scope required for '{tool}'.",
    "rate_limit": "Error 429: rate limit exceeded for '{tool}'. Retry after {secs} seconds.",
}

SIM_SYSTEM = (
    "You simulate the backend of a software tool exposed over MCP. Given the tool definition and the "
    "arguments of a call, reply with ONLY the raw result the real tool would return: realistic, internally "
    "consistent, specific values (ids, timestamps, names), consistent with the arguments. Prefer compact "
    "JSON. Never mention that this is a simulation. No markdown fences, no commentary."
)


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")
        self.db.commit()

    def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM responses WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, value))
        self.db.commit()


class MockBackend:
    """Server logic, independent of the MCP transport."""

    def __init__(
        self,
        tools: list[dict[str, Any]],
        cfg: dict[str, Any],
        *,
        error_rate: float | None = None,
        seed: int | None = None,
        offline: bool = False,
        cache_path: Path | None = None,
    ):
        self.tools = {t["name"]: t for t in tools}
        self.cfg = cfg
        mcfg = cfg["mock_server"]
        self.error_rate = mcfg["base_error_rate"] if error_rate is None else error_rate
        self.error_types = mcfg["error_types"]
        self.retry_success = mcfg["transient_retry_success"]
        self.deterministic = mcfg["deterministic"]
        self.seed = cfg["generation"]["seed"] if seed is None else seed
        self.offline = offline
        self.cache = None if offline else ResponseCache(cache_path or resolve(cfg["paths"]["mock_cache"]))
        self.attempts: dict[str, int] = {}
        self.sticky_error: dict[str, str] = {}  # call_key -> error kind injected on first attempt
        self._http: httpx.AsyncClient | None = None

    def list_tools(self) -> list[types.Tool]:
        return [
            types.Tool(name=t["name"], description=t.get("description") or "", inputSchema=t["inputSchema"])
            for t in self.tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return _error(f"Error: unknown tool '{name}'.")

        errors = sorted(validators.validator_for(tool["inputSchema"])(tool["inputSchema"]).iter_errors(arguments),
                        key=lambda e: list(e.path))
        if errors:
            details = "; ".join(_describe(e) for e in errors[:3])
            return _error(f"Error -32602: invalid params for '{name}': {details}")

        call_key = hashlib.sha256(_canon([self.seed, name, arguments]).encode()).hexdigest()
        attempt = self.attempts.get(call_key, 0)
        self.attempts[call_key] = attempt + 1
        injected = self._pick_error(call_key, attempt)
        if injected:
            rng = random.Random(f"{call_key}:{attempt}:msg")
            return _error(ERROR_TEMPLATES[injected].format(
                tool=name, secs=rng.choice([5, 10, 30, 60]), hint=_not_found_hint(arguments)))

        return types.CallToolResult(content=[types.TextContent(type="text", text=await self._simulate(tool, arguments, call_key))])

    def _pick_error(self, call_key: str, attempt: int) -> str | None:
        rng = random.Random(f"{call_key}:{attempt}") if self.deterministic else random.Random()
        if attempt == 0:
            if rng.random() >= self.error_rate:
                return None
            kinds, weights = zip(*self.error_types.items())
            self.sticky_error[call_key] = rng.choices(kinds, weights=weights)[0]
            return self.sticky_error[call_key]
        kind = self.sticky_error.get(call_key)
        # Persistent errors stay; transient ones usually clear on retry.
        if kind is None or (kind in TRANSIENT and rng.random() < self.retry_success):
            self.sticky_error.pop(call_key, None)
            return None
        return kind

    async def _simulate(self, tool: dict[str, Any], arguments: dict[str, Any], call_key: str) -> str:
        if self.offline:
            return _canon({"status": "ok", "tool": tool["name"], "arguments": arguments})
        cached = self.cache.get(call_key) if self.deterministic else None
        if cached is not None:
            return cached
        m = self.cfg["models"]["cheap"]
        body = {
            "model": m["slug"],
            "messages": [
                {"role": "system", "content": SIM_SYSTEM},
                {"role": "user", "content": (
                    f"Tool definition:\n{_canon({k: tool.get(k) for k in ('name', 'description', 'inputSchema')})}\n\n"
                    f"Call arguments:\n{_canon(arguments)}"
                )},
            ],
            "temperature": 0.0 if self.deterministic else m["temperature"],
            "max_tokens": m["max_tokens"],
            **m.get("extra_body", {}),
        }
        if self.deterministic:
            body["seed"] = self.seed
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.cfg["openrouter"]["endpoint"],
                headers={"Authorization": f"Bearer {api_key(self.cfg)}"},
                timeout=m["timeout"],
            )
        r = await self._http.post("/chat/completions", json=body)
        r.raise_for_status()
        text = _strip_fences(r.json()["choices"][0]["message"]["content"] or "")
        if self.deterministic and text:
            self.cache.put(call_key, text)
        return text


def _error(msg: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=msg)], isError=True)


def _describe(err: Any) -> str:
    where = "/".join(str(p) for p in err.absolute_path)
    return f"{where}: {err.message}" if where else err.message


def _not_found_hint(arguments: dict[str, Any]) -> str:
    for k, v in arguments.items():
        if isinstance(v, (str, int)) and re.search(r"id|name|path|key|email", k, re.I):
            return f"{k}={v!r}"
    return "no matching record"


def _strip_fences(text: str) -> str:
    m = re.match(r"^\s*```[a-zA-Z]*\n(.*?)\n```\s*$", text, re.S)
    return (m.group(1) if m else text).strip()


def build_server(backend: MockBackend) -> Server:
    server = Server("mock-mcp")

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        return backend.list_tools()

    # validate_input=False: we return our own, more realistic validation errors.
    @server.call_tool(validate_input=False)
    async def _call(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        try:
            return await backend.call(name, arguments or {})
        except Exception as exc:  # backend failure should look like a tool failure, not crash the episode
            return _error(f"Error: internal server error ({type(exc).__name__}).")

    return server


async def _serve(backend: MockBackend) -> None:
    server = build_server(backend)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--toolset", required=True, help="JSON file: list of {name, description, inputSchema, ...}")
    p.add_argument("--config", default=None)
    p.add_argument("--error-rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--offline", action="store_true", help="stub responses, no API calls")
    a = p.parse_args()
    cfg = load_config(a.config)
    tools = json.loads(Path(a.toolset).read_text(encoding="utf-8"))
    backend = MockBackend(tools, cfg, error_rate=a.error_rate, seed=a.seed, offline=a.offline)
    anyio.run(_serve, backend)


if __name__ == "__main__":
    main()
