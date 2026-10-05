"""Phase 1 step 1: collect real MCP tool schemas from public MCP servers.

Each server is launched locally over stdio (npx / uvx) and asked for `tools/list`; nothing else
is called. Dummy credentials are supplied where a server refuses to start without them. Only
servers whose package license is verified permissive are kept, and every tool records its
source package, version and license.

    python tools/collect_real_tools.py [--only name1,name2] [--refresh]
writes tools/library/real/<server>.jsonl (one file per server, so reruns skip finished servers)
and tools/library/real/_report.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import anyio
import httpx
from jsonschema import validators
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OUT = Path(__file__).resolve().parent / "library" / "real"
PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD", "Unlicense", "MIT-0"}
DUMMY = "dummy-not-a-real-key"
SANDBOX = Path(tempfile.gettempdir()) / "mcp_collect_sandbox"


def npm(pkg: str, domain: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> dict:
    # Scope is kept in the name: many vendors publish as `@vendor/mcp`.
    return {"name": pkg.lstrip("@").replace("/", "__"), "kind": "npm", "package": pkg, "domain": domain,
            "command": "npx", "args": ["-y", pkg, *(args or [])], "env": env or {}}


def pypi(pkg: str, domain: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> dict:
    return {"name": pkg, "kind": "pypi", "package": pkg, "domain": domain,
            "command": "uvx", "args": [pkg, *(args or [])], "env": env or {}}


SERVERS = [
    # Official reference servers
    npm("@modelcontextprotocol/server-filesystem", "files", [str(SANDBOX)]),
    npm("@modelcontextprotocol/server-memory", "knowledge_graph"),
    npm("@modelcontextprotocol/server-everything", "demo"),
    npm("@modelcontextprotocol/server-sequential-thinking", "reasoning"),
    npm("@modelcontextprotocol/server-github", "devtools", env={"GITHUB_PERSONAL_ACCESS_TOKEN": DUMMY}),
    npm("@modelcontextprotocol/server-gitlab", "devtools", env={"GITLAB_PERSONAL_ACCESS_TOKEN": DUMMY}),
    npm("@modelcontextprotocol/server-slack", "messaging", env={"SLACK_BOT_TOKEN": DUMMY, "SLACK_TEAM_ID": "T000"}),
    npm("@modelcontextprotocol/server-brave-search", "search", env={"BRAVE_API_KEY": DUMMY}),
    npm("@modelcontextprotocol/server-google-maps", "maps", env={"GOOGLE_MAPS_API_KEY": DUMMY}),
    npm("@modelcontextprotocol/server-postgres", "database", ["postgresql://localhost:1/none"]),
    npm("@modelcontextprotocol/server-redis", "database", ["redis://localhost:1"]),
    npm("@modelcontextprotocol/server-aws-kb-retrieval", "search",
        env={"AWS_ACCESS_KEY_ID": DUMMY, "AWS_SECRET_ACCESS_KEY": DUMMY, "AWS_REGION": "us-east-1"}),
    npm("@modelcontextprotocol/server-everart", "image_gen", env={"EVERART_API_KEY": DUMMY}),
    pypi("mcp-server-git", "devtools", ["--repository", str(SANDBOX)]),
    pypi("mcp-server-fetch", "search"),
    pypi("mcp-server-time", "time"),
    # Vendor servers
    npm("@notionhq/notion-mcp-server", "notes", env={"NOTION_TOKEN": DUMMY}),
    npm("@playwright/mcp", "browser"),
    npm("@stripe/mcp", "payments", ["--tools=all", f"--api-key=sk_test_{DUMMY}"]),
    npm("@supabase/mcp-server-supabase", "database", [f"--access-token={DUMMY}"]),
    npm("@upstash/context7-mcp", "docs"),
    npm("firecrawl-mcp", "search", env={"FIRECRAWL_API_KEY": DUMMY}),
    npm("@browserbasehq/mcp", "browser", env={"BROWSERBASE_API_KEY": DUMMY, "BROWSERBASE_PROJECT_ID": DUMMY}),
    npm("@heroku/mcp-server", "cloud", env={"HEROKU_API_KEY": DUMMY}),
    npm("mcp-server-kubernetes", "cloud"),
    npm("@azure/mcp", "cloud", ["server", "start"]),
    npm("@hubspot/mcp-server", "crm", env={"PRIVATE_APP_ACCESS_TOKEN": DUMMY}),
    npm("@twilio-alpha/mcp", "messaging", ["ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/SKxxxxxxxx:secret"]),
    npm("@shopify/dev-mcp", "ecommerce"),
    npm("@elastic/mcp-server-elasticsearch", "database",
        env={"ES_URL": "http://localhost:1", "ES_API_KEY": DUMMY}),
    npm("@neondatabase/mcp-server-neon", "database", ["start", DUMMY]),
    npm("tavily-mcp", "search", env={"TAVILY_API_KEY": DUMMY}),
    npm("@kimtaeyoon83/mcp-server-youtube-transcript", "media"),
    npm("mongodb-mcp-server", "database"),
    npm("@circleci/mcp-server-circleci", "devtools", env={"CIRCLECI_TOKEN": DUMMY}),
    npm("@apify/actors-mcp-server", "automation", env={"APIFY_TOKEN": DUMMY}),
    npm("@mastra/mcp-docs-server", "docs"),
    npm("@raygun.io/mcp-server-raygun", "observability", env={"RAYGUN_PAT_TOKEN": DUMMY}),
    npm("@pinecone-database/mcp", "database", env={"PINECONE_API_KEY": DUMMY}),
    npm("@dynatrace-oss/dynatrace-mcp-server", "observability",
        env={"DT_ENVIRONMENT": "https://abc123.apps.dynatrace.com"}),
    npm("@chargebee/mcp", "payments"),
    npm("@adyen/mcp", "payments", ["--adyenApiKey", DUMMY, "--env", "TEST"]),
    pypi("awslabs.aws-documentation-mcp-server", "docs"),
    pypi("mcp-server-qdrant", "database", env={"QDRANT_URL": "http://localhost:1", "COLLECTION_NAME": "x"}),
    pypi("arxiv-mcp-server", "search"),
    pypi("awslabs.dynamodb-mcp-server", "database"),
]


# ---------------------------------------------------------------------------------------------
# License checks
# ---------------------------------------------------------------------------------------------


def _classify_license_text(text: str) -> str | None:
    t = text[:4000]
    if "Permission is hereby granted, free of charge" in t:
        return "MIT"
    if "Apache License" in t and "Version 2.0" in t:
        return "Apache-2.0"
    if "Redistribution and use in source and binary forms" in t:
        return "BSD-3-Clause" if "Neither the name" in t else "BSD-2-Clause"
    if "Permission to use, copy, modify, and/or distribute" in t:
        return "ISC"
    return None


def resolve_license(server: dict) -> tuple[str | None, str | None, str]:
    """Returns (version, spdx_license_or_None, raw_license_field)."""
    pkg = server["package"]
    if server["kind"] == "npm":
        r = httpx.get(f"https://registry.npmjs.org/{pkg.replace('/', '%2F')}/latest", timeout=30)
        if r.status_code != 200:
            return None, None, f"http {r.status_code}"
        meta = r.json()
        version, raw = meta.get("version"), meta.get("license")
        if isinstance(raw, dict):
            raw = raw.get("type")
        if raw in PERMISSIVE:
            return version, raw, raw
        m = re.match(r"SEE LICENSE IN (\S+)", raw or "")
        if m:  # read the license file shipped in the package
            f = httpx.get(f"https://unpkg.com/{pkg}@{version}/{m.group(1)}", timeout=30, follow_redirects=True)
            return version, _classify_license_text(f.text) if f.status_code == 200 else None, raw
        return version, None, str(raw)
    r = httpx.get(f"https://pypi.org/pypi/{pkg}/json", timeout=30)
    if r.status_code != 200:
        return None, None, f"http {r.status_code}"
    info = r.json()["info"]
    raw = info.get("license_expression") or info.get("license") or ""
    if raw in PERMISSIVE:
        return info["version"], raw, raw
    for c in info.get("classifiers", []):
        if "MIT License" in c:
            return info["version"], "MIT", c
        if "Apache Software License" in c:
            return info["version"], "Apache-2.0", c
    return info["version"], _classify_license_text(raw), raw[:60]


# ---------------------------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------------------------


async def _list_tools(server: dict, timeout: float, errlog) -> list[dict]:
    params = StdioServerParameters(command=server["command"], args=server["args"], env=server["env"] or None,
                                   cwd=str(SANDBOX))
    with anyio.fail_after(timeout):
        async with stdio_client(params, errlog=errlog) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            tools, cursor = [], None
            while True:
                page = await s.list_tools(cursor=cursor) if cursor else await s.list_tools()
                tools += [t.model_dump(exclude_none=True) for t in page.tools]
                cursor = page.nextCursor
                if not cursor:
                    return tools


def normalize(tool: dict, server: dict, version: str, license_: str) -> dict | None:
    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    try:
        validators.validator_for(schema).check_schema(schema)
    except Exception:
        return None
    return {
        "id": f"real/{server['name']}/{tool['name']}",
        "name": tool["name"],
        "description": tool.get("description") or "",
        "inputSchema": schema,
        "domain": server["domain"],
        "source": f"{server['kind']}:{server['package']}@{version}",
        "license": license_,
    }


def collect_one(server: dict, timeout: float, refresh: bool) -> dict:
    out_path = OUT / f"{server['name']}.jsonl"
    if out_path.exists() and not refresh:
        return {"server": server["name"], "status": "cached", "tools": sum(1 for _ in out_path.open())}
    try:
        version, license_, raw = resolve_license(server)
    except Exception as exc:
        return {"server": server["name"], "status": f"license lookup failed: {exc}"}
    if license_ not in PERMISSIVE:
        return {"server": server["name"], "status": f"skipped: license not verified permissive ({raw})"}
    log_path = OUT / "_logs" / f"{server['name']}.log"
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as errlog:
            tools = anyio.run(_list_tools, server, timeout, errlog)
    except BaseException as exc:  # noqa: BLE001 - any startup failure just skips the server
        return {"server": server["name"], "status": f"failed: {type(exc).__name__}: {str(exc)[:150]}"}
    rows = [r for t in tools if (r := normalize(t, server, version, license_))]
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"server": server["name"], "status": "ok", "tools": len(rows), "invalid": len(tools) - len(rows),
            "license": license_, "version": version}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, help="comma-separated server names")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--timeout", type=float, default=180.0, help="per server, includes first-run download")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    SANDBOX.mkdir(parents=True, exist_ok=True)
    if not (SANDBOX / ".git").exists():  # mcp-server-git needs a repository to point at
        subprocess.run(["git", "init", "-q", str(SANDBOX)], check=False)
    (OUT / "_logs").mkdir(parents=True, exist_ok=True)
    servers = [s for s in SERVERS if not a.only or s["name"] in a.only.split(",")]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda s: collect_one(s, a.timeout, a.refresh), servers))
    for r in results:
        print(f"{r['server']:40s} {r['status']:10.80s} {r.get('tools', '')}")
    total = sum(r.get("tools", 0) for r in results if r["status"] in ("ok", "cached"))
    print(f"\n{total} tools from {sum(r['status'] in ('ok', 'cached') for r in results)}/{len(results)} servers")
    (OUT / "_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
