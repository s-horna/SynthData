"""Shared helpers: config loading, .env, OpenRouter metadata and spend tracking."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def api_key(cfg: dict[str, Any]) -> str:
    key = os.environ.get(cfg["openrouter"]["api_key_env"])
    if not key:
        raise SystemExit(f"{cfg['openrouter']['api_key_env']} is not set (put it in .env).")
    return key


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def fetch_models(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    r = httpx.get(f"{cfg['openrouter']['endpoint']}/models", timeout=30)
    r.raise_for_status()
    return {m["id"]: m for m in r.json()["data"]}


def key_usage_usd(cfg: dict[str, Any]) -> float:
    """Total credits used by this API key, as reported by OpenRouter."""
    r = httpx.get(
        f"{cfg['openrouter']['endpoint']}/key",
        headers={"Authorization": f"Bearer {api_key(cfg)}"},
        timeout=30,
    )
    r.raise_for_status()
    return float(r.json()["data"]["usage"])


class SpendLedger:
    """Append-only per-run ledger of measured spend, so caps survive resumes.

    Spend is measured as the delta in the key's OpenRouter usage around each shard,
    which covers the teacher, the user-request model and the mock server's calls.
    """

    def __init__(self, path: Path, cfg: dict[str, Any]):
        self.path = path
        self.cfg = cfg
        self.entries: list[dict[str, Any]] = []
        if path.exists():
            self.entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    @property
    def spent(self) -> float:
        return sum(e["usd"] for e in self.entries)

    @property
    def rows(self) -> int:
        return sum(e["rows"] for e in self.entries)

    def cost_per_row(self) -> float | None:
        return self.spent / self.rows if self.rows else None

    def record(self, shard: str, before: float, rows: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        time.sleep(self.cfg["budget"]["settle_seconds"])
        after = key_usage_usd(self.cfg)
        entry = {"shard": shard, "usd": round(after - before, 6), "rows": rows, "ts": time.time(), **(extra or {})}
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry
