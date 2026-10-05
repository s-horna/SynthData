"""LLM-backed mock MCP server (Phase 2).

Exposes whatever toolset it is given and answers calls with the cheap model:
  - arguments are validated with jsonschema; invalid ones get a realistic MCP error
  - an optional "world": backend records that must exist (the items the users' requests refer to),
    which the simulator includes wherever a call would return them
  - seeded, deterministic mode with an on-disk response cache

Run as a stdio MCP server (this is how Data Designer launches it):
    python -m mock_server.server --toolset path/to/toolset.json [--world records.json] [--salt KEY] [--offline]
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

SIM_SYSTEM = (
    "You simulate the backend of a software tool exposed over MCP. Given the tool definition and the "
    "arguments of a call, reply with ONLY the raw result the real tool would return: realistic, internally "
    "consistent, specific values (ids, timestamps, names), consistent with the arguments. Prefer compact "
    "JSON. Keep list results to a handful of items. If the tool is paginated, return a single complete page: "
    "no next-page cursor or token, and has_more/hasMore false if such a field exists. Never mention that this "
    "is a simulation. No markdown fences, no commentary. Use varied, domain-specific values; never placeholders like "
    "example.com, foo/bar, abc123, Alice/Bob or John Doe."
)

WORLD_INSTRUCTIONS = (
    "These records exist in the backend. Whenever this call would return or act on one of them (a list, search or "
    "get that matches it), include it with exactly these ids and details, alongside other realistic records where "
    "a list would have them. Never contradict them. If the call matches none of them, ignore this section."
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
        seed: int | None = None,
        world: list[str] | None = None,
        salt: str = "",
        offline: bool = False,
        cache_path: Path | None = None,
    ):
        self.tools = {t["name"]: t for t in tools}
        self.cfg = cfg
        self.deterministic = cfg["mock_server"]["deterministic"]
        self.seed = cfg["generation"]["seed"] if seed is None else seed
        self.world = world or []
        self.salt = salt
        self.offline = offline
        self.cache = None if offline else ResponseCache(cache_path or resolve(cfg["paths"]["mock_cache"]))
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

        # Prompts, world and salt are part of the key: cached responses are regenerated when any of them change,
        # and the same call in another slot gets its own response.
        call_key = hashlib.sha256(_canon([self.seed, self.salt, SIM_SYSTEM, WORLD_INSTRUCTIONS, self.world, name,
                                          arguments]).encode()).hexdigest()
        return types.CallToolResult(content=[types.TextContent(type="text", text=await self._simulate(tool, arguments, call_key))])

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
                    + (WORLD_INSTRUCTIONS + "\n" + "\n".join(f"- {r}" for r in self.world) + "\n\n" if self.world else "")
                    + f"Call arguments:\n{_canon(arguments)}"
                )},
            ],
            "temperature": 0.0 if self.deterministic else m["temperature"],
            "max_tokens": m["max_tokens"],
            **m.get("extra_body", {}),
        }
        if self.deterministic:
            body["seed"] = int(call_key[:8], 16)
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
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--world", default=None, help="JSON file: list of backend record descriptions")
    p.add_argument("--salt", default="", help="extra cache-key component, e.g. run/shard/slot")
    p.add_argument("--offline", action="store_true", help="stub responses, no API calls")
    a = p.parse_args()
    cfg = load_config(a.config)
    tools = json.loads(Path(a.toolset).read_text(encoding="utf-8"))
    world = json.loads(Path(a.world).read_text(encoding="utf-8")) if a.world else None
    backend = MockBackend(tools, cfg, seed=a.seed, world=world, salt=a.salt, offline=a.offline)
    anyio.run(_serve, backend)


if __name__ == "__main__":
    main()
