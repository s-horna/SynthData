"""Phase 3: generate tool-calling trajectories with NeMo Data Designer.

Each row gets a toolset (3-20 train tools, sometimes 30-60, with look-alike distractors) and a
scenario; the cheap model writes the user request and the teacher solves it against the mock
MCP server via `tool_alias`, with full message traces.

Data Designer caches an MCP provider's tool list, so a toolset can't vary per row through a
single provider. Instead each *shard* (one Data Designer job) launches N mock-server "slots",
one per toolset, each with its own ToolConfig. Every row is routed to its slot's trajectory
column via `skip`, and the columns are merged afterwards.

Commands (from the repo root):
    python -m generate.generate check [--test-calls]   # Checkpoint 1: slugs, context, prices
    python -m generate.generate preview [--n 50]       # ~50 rows, then stop for manual review
    python -m generate.generate run --n 500 --run-name pilot         # pilot, under the pilot cap
    python -m generate.generate run --full --run-name full --yes     # full run, after Checkpoint 2

Runs are resumable: finished shards are skipped, interrupted shards resume inside Data Designer.
Output: data/raw/<run>/shards/shard_XXXX.parquet (+ toolsets/, spend in data/raw/spend_ledger.jsonl).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, SpendLedger, api_key, fetch_models, key_usage_usd, load_config, resolve  # noqa: E402

TEACHER_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Use the provided tools when they are needed to "
    "fulfil the user's request, and call them with arguments that exactly follow their schemas. Never invent "
    "values for required parameters: if required information is missing or ambiguous, ask the user a short "
    "clarifying question instead of calling a tool. If none of the available tools can do what the user "
    "asks, say so plainly and do not call any tool. Independent calls may be made in parallel. If a tool "
    "returns an error, decide whether to retry, try an alternative, or explain the failure. Finish with a "
    "concise answer grounded in the tool results."
)

SCENARIO_INSTRUCTIONS = {
    "single_call": "The request should be fully solvable with exactly ONE call to one of the tools, and include "
    "every value that tool's required parameters need.",
    "sequential_multi": "The request should need 2-4 tool calls in sequence, where a later call depends on the "
    "output of an earlier one (e.g. look something up, then act on the result).",
    "parallel": "The request should need 2-4 INDEPENDENT tool calls that can run at the same time (e.g. the same "
    "lookup for several items, or unrelated actions in one message).",
    "no_tool_fits": "The request must be plausible for someone using these tools, close to their domain, but NOT "
    "achievable with any of them. Do not ask for something trivially unrelated.",
    "needs_clarification": "The request should clearly target one of the tools but leave out a value for one of its "
    "REQUIRED parameters that cannot be inferred (say which in missing_info), so a careful assistant must ask.",
    "error_recovery": "The request should need 1-3 tool calls with all required values present. (The backend may "
    "fail; the request itself should be normal.)",
}

USER_REQUEST_PROMPT = """You are writing a realistic message that a user sends to an AI assistant which has access to the tools below.

Available tools:
{{ toolset_brief }}

User persona: {{ persona }}
Phrasing style: {{ phrasing_style }}
Difficulty: {{ difficulty }}

Scenario: {{ scenario_instructions }}

Rules:
- Write in the persona's voice and the requested phrasing style ("typos" = casual with a few spelling mistakes; "non-native English" = small grammar slips).
- Use natural language. Never mention tool names, function names, parameter names, or JSON.
- Include concrete, realistic values (names, dates, ids, amounts) where the scenario requires them.
- Harder difficulty = more implicit intent, more steps, or details spread across the message.

Return `request` (the user's message only), `intended_tools` (tool names a perfect assistant would call, empty if none), and `missing_info` (what is deliberately missing, or null)."""


class UserRequest(BaseModel):
    request: str = Field(description="The user's message to the assistant, verbatim.")
    intended_tools: list[str] = Field(default_factory=list, description="Tool names a perfect assistant would call.")
    missing_info: str | None = Field(default=None, description="Deliberately missing required info, if any.")


# --------------------------------------------------------------------------------------------
# Tool library and toolset sampling
# --------------------------------------------------------------------------------------------


def load_train_tools(cfg: dict[str, Any], tools_path: Path | None = None, split_path: Path | None = None) -> list[dict]:
    tools_path = tools_path or resolve(cfg["paths"]["tool_library"])
    split_path = split_path or resolve(cfg["paths"]["domain_split"])
    if not tools_path.exists() or not split_path.exists():
        raise SystemExit(f"Tool library not found ({tools_path}, {split_path}). Run Phase 1 first.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train, heldout = set(split["train_domains"]), set(split["heldout_domains"])
    if train & heldout:
        raise SystemExit(f"Domains in both train and held-out: {sorted(train & heldout)}")

    # Different servers can share a tool name (e.g. two `create_issue`s); all are kept here and the
    # sampler keeps names unique within each toolset.
    tools, seen = [], set()
    heldout_names = set()
    for line in tools_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if t["domain"] in heldout:
            heldout_names.add(t["name"])
        elif t["domain"] in train:
            seen.add(t["name"])
            tools.append(t)
    leaked = seen & heldout_names
    if leaked:  # a train tool sharing a name with a held-out tool would leak it into training data
        tools = [t for t in tools if t["name"] not in leaked]
        print(f"[warn] dropped {len(leaked)} train tools whose names collide with held-out tools")
    if not tools:
        raise SystemExit("No train-domain tools available.")
    return tools


def _name_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[_\-.\s]+|(?<=[a-z])(?=[A-Z])", name.lower()) if t}


class ToolsetSampler:
    def __init__(self, tools: list[dict], cfg: dict[str, Any]):
        self.tools = tools
        self.tcfg = cfg["generation"]["toolset"]
        self.by_domain: dict[str, list[dict]] = defaultdict(list)
        for t in tools:
            self.by_domain[t["domain"]].append(t)
        self.domains = sorted(self.by_domain)
        self.tokens = {t["name"]: _name_tokens(t["name"]) for t in tools}

    def sample(self, rng: random.Random) -> list[dict]:
        lo, hi = self.tcfg["large_range"] if rng.random() < self.tcfg["large_prob"] else self.tcfg["small_range"]
        size = min(rng.randint(lo, hi), len(self.tools))

        domain = rng.choice(self.domains)
        pool = self.by_domain[domain]
        core = rng.sample(pool, min(len(pool), rng.randint(1, min(4, size))))
        chosen = {t["name"]: t for t in core}

        # Look-alike distractors: same domain or overlapping name tokens with the core tools.
        n_distract = round((size - len(chosen)) * self.tcfg["distractor_frac"])
        core_tokens = set().union(*(self.tokens[t["name"]] for t in core))
        candidates = [
            t for t in self.tools
            if t["name"] not in chosen and (t["domain"] == domain or self.tokens[t["name"]] & core_tokens)
        ]
        candidates.sort(key=lambda t: (-len(self.tokens[t["name"]] & core_tokens), rng.random()))
        top = candidates[: max(n_distract * 3, n_distract)]
        for t in rng.sample(top, min(n_distract, len(top))):
            chosen[t["name"]] = t

        while len(chosen) < size:
            t = rng.choice(self.tools)
            chosen.setdefault(t["name"], t)

        out = list(chosen.values())
        rng.shuffle(out)
        return out


def toolset_brief(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        schema = t["inputSchema"]
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        params = ", ".join(f"{k}{'*' if k in required else ''}" for k in props) or "no parameters"
        desc = (t.get("description") or "(no description)").strip().replace("\n", " ")
        lines.append(f"- {t['name']}: {desc[:300]} [params: {params}]")
    return "\n".join(lines) + "\n(* = required)"


# --------------------------------------------------------------------------------------------
# Shard planning
# --------------------------------------------------------------------------------------------


def plan_shard(
    sampler: ToolsetSampler, cfg: dict[str, Any], run: str, shard_idx: int, n_rows: int
) -> tuple[pd.DataFrame, list[dict]]:
    """Deterministically build one shard's seed rows and slot definitions."""
    g = cfg["generation"]
    rng = random.Random(f"{g['seed']}:{run}:{shard_idx}")
    per_slot = g["rows_per_slot"]
    n_slots = min(g["slots_per_shard"], math.ceil(n_rows / per_slot))

    weights = dict(g["scenario_weights"])
    err_w = weights.pop("error_recovery", 0.0)
    total_w = err_w + sum(weights.values())
    n_err_slots = 0
    if err_w > 0 and n_slots > 1:
        n_err_slots = max(1, round(n_slots * err_w / total_w))
    elif err_w > 0 and rng.random() < err_w / total_w:
        n_err_slots = 1

    slots, rows = [], []
    for k in range(n_slots):
        tools = sampler.sample(rng)
        is_err = k < n_err_slots
        slots.append({
            "slot": k,
            "tools": tools,
            "error_rate": cfg["mock_server"]["error_recovery_error_rate" if is_err else "base_error_rate"],
        })
        for _ in range(per_slot):
            if len(rows) >= n_rows:
                break
            scenario = "error_recovery" if is_err else rng.choices(list(weights), weights=list(weights.values()))[0]
            rows.append({
                "slot": k,
                "scenario_type": scenario,
                "scenario_instructions": SCENARIO_INSTRUCTIONS[scenario],
                "toolset_brief": toolset_brief(tools),
                "toolset": json.dumps(tools, ensure_ascii=False),
                "toolset_size": len(tools),
                "primary_domain": _primary_domain(tools),
                "source_tool_ids": json.dumps([t.get("id", t["name"]) for t in tools]),
            })
    rng.shuffle(rows)
    return pd.DataFrame(rows), slots


def _primary_domain(tools: list[dict]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for t in tools:
        counts[t["domain"]] += 1
    return max(counts, key=counts.get)


# --------------------------------------------------------------------------------------------
# Data Designer config
# --------------------------------------------------------------------------------------------


def model_configs(cfg: dict[str, Any], roles: list[str], *, health_check: bool):
    import data_designer.config as dd

    out = []
    for role in roles:
        m = cfg["models"][role]
        out.append(dd.ModelConfig(
            alias=m["alias"],
            model=m["slug"],
            provider="openrouter",
            skip_health_check=not health_check,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=m["temperature"],
                max_tokens=m["max_tokens"],
                max_parallel_requests=m["max_parallel_requests"],
                timeout=m["timeout"],
                extra_body=m.get("extra_body") or None,
            ),
        ))
    return out


def openrouter_provider(cfg: dict[str, Any]):
    import data_designer.config as dd

    # api_key holds the env var NAME; Data Designer's secret resolver reads the value from the env.
    return dd.ModelProvider(name="openrouter", endpoint=cfg["openrouter"]["endpoint"],
                            api_key=cfg["openrouter"]["api_key_env"])


def build_shard_config(cfg: dict[str, Any], seed_df: pd.DataFrame, slots: list[dict], *, health_check: bool):
    import data_designer.config as dd

    g = cfg["generation"]
    teacher, cheap = cfg["models"]["teacher"]["alias"], cfg["models"]["cheap"]["alias"]
    b = dd.DataDesignerConfigBuilder(model_configs=model_configs(cfg, ["teacher", "cheap"], health_check=health_check))
    b.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df), sampling_strategy=dd.SamplingStrategy.ORDERED)

    for name, values, weights in [
        ("persona", g["personas"], None),
        ("phrasing_style", g["phrasing_styles"], g["phrasing_weights"]),
        ("difficulty", g["difficulties"], g["difficulty_weights"]),
    ]:
        b.add_column(dd.SamplerColumnConfig(
            name=name, sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=values, weights=weights),
        ))

    b.add_column(dd.LLMStructuredColumnConfig(
        name="user_request", prompt=USER_REQUEST_PROMPT, output_format=UserRequest, model_alias=cheap,
    ))

    for s in slots:
        k = s["slot"]
        b.add_tool_config(dd.ToolConfig(
            tool_alias=f"slot{k}", providers=[f"mock-slot{k}"],
            max_tool_call_turns=g["max_tool_call_turns"], timeout_sec=g["tool_timeout_sec"],
        ))
        b.add_column(dd.LLMTextColumnConfig(
            name=f"trajectory_s{k}",
            system_prompt=TEACHER_SYSTEM_PROMPT,
            prompt="{{ user_request.request }}",
            model_alias=teacher,
            tool_alias=f"slot{k}",
            with_trace=dd.TraceType.ALL_MESSAGES,
            # Render to '' (falsy) for this slot's rows; any non-empty string means skip.
            skip=dd.SkipConfig(when=f"{{{{ 'skip' if slot != {k} else '' }}}}"),
        ))
    return b


def mcp_providers(cfg: dict[str, Any], slots: list[dict], toolset_dir: Path, shard: str, *, offline: bool):
    import data_designer.config as dd

    cfg_key_env = cfg["openrouter"]["api_key_env"]
    api_key_value = None if offline else api_key(cfg)

    providers = []
    for s in slots:
        path = toolset_dir / f"{shard}_slot{s['slot']:02d}.json"
        path.write_text(json.dumps(s["tools"], ensure_ascii=False), encoding="utf-8")
        args = ["-m", "mock_server.server", "--toolset", str(path), "--error-rate", str(s["error_rate"])]
        if offline:
            args.append("--offline")
        # The MCP SDK spawns servers with a minimal whitelisted env, so the key must be passed explicitly.
        # (Providers are runtime-only; Data Designer doesn't write them to its artifacts.)
        env = {"PYTHONPATH": str(ROOT)}
        if not offline:
            env[cfg_key_env] = api_key_value
        providers.append(dd.LocalStdioMCPProvider(
            name=f"mock-slot{s['slot']}", command=sys.executable, args=args, env=env,
        ))
    return providers


# --------------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------------


def merge_slots(df: pd.DataFrame, slots: list[dict], cfg: dict[str, Any], run: str, shard: str) -> pd.DataFrame:
    from data_designer.engine.mcp.facade import DEFAULT_TOOL_REFUSAL_MESSAGE

    final, traces = [], []
    for _, row in df.iterrows():
        k = int(row["slot"])
        final.append(row.get(f"trajectory_s{k}"))
        traces.append(row.get(f"trajectory_s{k}__trace"))
    slot_cols = [c for c in df.columns if re.fullmatch(r"trajectory_s\d+(__trace)?", c)]
    out = df.drop(columns=slot_cols + ["scenario_instructions", "toolset_brief"], errors="ignore").copy()
    out["final_response"] = final
    out["trace"] = [json.dumps(_to_py(t), ensure_ascii=False) if _present(t) else None for t in traces]
    out["hit_max_turns"] = [bool(t) and DEFAULT_TOOL_REFUSAL_MESSAGE in t for t in out["trace"]]
    out["user_request"] = [json.dumps(_to_py(u), ensure_ascii=False) for u in out["user_request"]]
    t, c = cfg["models"]["teacher"], cfg["models"]["cheap"]
    out["teacher_model"] = t["slug"]
    out["teacher_license"] = t["license"]
    out["user_model"] = c["slug"]
    out["user_model_license"] = c["license"]
    out["mock_model"] = c["slug"]
    out["run"] = run
    out["shard"] = shard
    out["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _to_py(v: Any, top: bool = True) -> Any:
    """Convert numpy/arrow containers from parquet round-trips back into plain Python.

    Only a top-level JSON string is decoded: nested strings (tool results, tool-call `arguments`)
    must stay strings, as in the OpenAI message format.
    """
    if hasattr(v, "tolist"):
        v = v.tolist()
    if top and isinstance(v, str) and v[:1] in "[{":
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return v
    if isinstance(v, dict):
        return {k: _to_py(x, False) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_py(x, False) for x in v]
    return v


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


def require_confirmed(cfg: dict[str, Any]) -> None:
    if not cfg["models"].get("confirmed"):
        raise SystemExit("Model slugs are not confirmed (Checkpoint 1). Run `check`, then set models.confirmed: true.")


def run_generation(
    cfg: dict[str, Any], *, run: str, n: int, cap_usd: float, cap_scope: str, offline: bool = False,
    tools_path: Path | None = None, split_path: Path | None = None,
) -> Path:
    from data_designer.config import ResumeMode
    from data_designer.engine.mcp import io as mcp_io
    from data_designer.interface import DataDesigner

    raw = resolve(cfg["paths"]["raw"])
    run_dir = raw / run
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    (run_dir / "toolsets").mkdir(exist_ok=True)
    # Tracked even with --offline: the LLM columns still bill OpenRouter.
    ledger = SpendLedger(raw / "spend_ledger.jsonl", cfg)

    def scope_spent() -> float:
        return sum(e["usd"] for e in ledger.entries if e.get("scope") == cap_scope)

    def scope_cpr() -> float | None:
        es = [e for e in ledger.entries if e.get("scope") == cap_scope and e["rows"]]
        return sum(e["usd"] for e in es) / sum(e["rows"] for e in es) if es else None

    sampler = ToolsetSampler(load_train_tools(cfg, tools_path, split_path), cfg)
    shard_rows = cfg["generation"]["slots_per_shard"] * cfg["generation"]["rows_per_slot"]
    n_shards = math.ceil(n / shard_rows)
    print(f"[{run}] {n} rows -> {n_shards} shard(s) of <= {shard_rows}; {len(sampler.tools)} train tools, "
          f"{len(sampler.domains)} domains; cap ${cap_usd:.2f} ({cap_scope})")

    first = True
    for i in range(n_shards):
        shard = f"shard_{i:04d}"
        out_path = run_dir / "shards" / f"{shard}.parquet"
        if out_path.exists():
            continue
        rows_this = min(shard_rows, n - i * shard_rows)

        cpr = scope_cpr()
        est = cpr * rows_this if cpr else 0.0
        if scope_spent() + est > cap_usd:
            print(f"[stop] spent ${scope_spent():.2f} + est ${est:.2f} for {shard} would exceed cap ${cap_usd:.2f}")
            break

        seed_df, slots = plan_shard(sampler, cfg, run, i, rows_this)
        providers = mcp_providers(cfg, slots, run_dir / "toolsets", shard, offline=offline)
        builder = build_shard_config(cfg, seed_df, slots, health_check=first and not offline)
        designer = DataDesigner(artifact_path=run_dir / "_dd", model_providers=[openrouter_provider(cfg)],
                                mcp_providers=providers)
        before = key_usage_usd(cfg)
        try:
            res = designer.create(builder, num_records=len(seed_df), dataset_name=shard,
                                  resume=ResumeMode.IF_POSSIBLE)
            df = res.load_dataset()
        finally:
            mcp_io.clear_provider_caches(providers)  # stop this shard's mock-server subprocesses
            entry = ledger.record(f"{run}/{shard}", before, 0, {"scope": cap_scope, "run": run})
        first = False

        merged = merge_slots(df, slots, cfg, run, shard)
        merged.to_parquet(out_path, index=False)
        # Attribute rows to the entry just written (cost is recorded even if create failed).
        ledger.entries[-1]["rows"] = len(merged)
        _rewrite_ledger(ledger)
        print(f"[{shard}] {len(merged)}/{len(seed_df)} rows, ${entry['usd']:.4f}; "
              f"{cap_scope} total ${scope_spent():.2f}/{cap_usd:.2f}")
    return run_dir


def _rewrite_ledger(ledger: SpendLedger) -> None:
    ledger.path.write_text("".join(json.dumps(e) + "\n" for e in ledger.entries), encoding="utf-8")


def load_run(run_dir: Path) -> pd.DataFrame:
    files = sorted((run_dir / "shards").glob("shard_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def report(cfg: dict[str, Any], run_dir: Path) -> None:
    df = load_run(run_dir)
    print(f"\n=== {run_dir.name}: {len(df)} rows ===")
    if df.empty:
        return
    print(df["scenario_type"].value_counts().to_string())
    print(f"traces present: {df['trace'].notna().sum()}  hit max turns: {int(df['hit_max_turns'].sum())}")
    ledger_path = resolve(cfg["paths"]["raw"]) / "spend_ledger.jsonl"
    if not ledger_path.exists():
        return
    ledger = SpendLedger(ledger_path, cfg)
    es = [e for e in ledger.entries if e.get("run") == run_dir.name]
    spent, rows = sum(e["usd"] for e in es), sum(e["rows"] for e in es)
    if not rows:
        return
    b = cfg["budget"]
    cpr = spent / rows
    cpk = cpr / b["keep_rate_assumption"]
    print(f"spend ${spent:.4f} for {rows} rows -> ${cpr:.4f}/generated row")
    print(f"cost per kept example (ASSUMED keep rate {b['keep_rate_assumption']:.0%}; Phase 4 will measure it): "
          f"${cpk:.4f}")
    print(f"projected full run: {b['target_kept']} kept -> ~{math.ceil(b['target_kept'] / b['keep_rate_assumption'])}"
          f" generated -> ~${cpk * b['target_kept']:.2f}")
    pilot = sum(e["usd"] for e in ledger.entries if e.get("scope") == "pilot")
    print(f"pilot budget used: ${pilot:.2f} / ${b['pilot_usd']:.2f}")


def show_samples(run_dir: Path, k: int = 3) -> None:
    df = load_run(run_dir)
    for _, r in df.sample(min(k, len(df)), random_state=0).iterrows():
        print(f"\n--- {r['scenario_type']} | {r['primary_domain']} | {r['toolset_size']} tools ---")
        for m in json.loads(r["trace"] or "[]"):
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            if m["role"] == "system":
                continue
            print(f"[{m['role']}] {str(content)[:400]}")
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", {})
                print(f"    -> {f.get('name')}({str(f.get('arguments'))[:300]})")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def cmd_check(cfg: dict[str, Any], test_calls: bool) -> None:
    api_key(cfg)
    print("OPENROUTER_API_KEY: set")
    catalog = fetch_models(cfg)
    ok = True
    for role in ("teacher", "cheap", "judge"):
        m = cfg["models"][role]
        info = catalog.get(m["slug"])
        if info is None:
            ok = False
            print(f"{role:8s} {m['slug']}: NOT FOUND on OpenRouter")
            continue
        p = info.get("pricing", {})
        sp = info.get("supported_parameters") or []
        print(f"{role:8s} {m['slug']}: ctx={info.get('context_length')}  "
              f"in=${float(p.get('prompt', 0)) * 1e6:.3f}/M  out=${float(p.get('completion', 0)) * 1e6:.3f}/M  "
              f"tools={'yes' if 'tools' in sp else 'NO'}  reasoning={'yes' if 'reasoning' in sp else 'no'}")
        if role == "teacher" and "tools" not in sp:
            ok = False
    print(f"key usage so far: ${key_usage_usd(cfg):.4f}")
    print(f"models.confirmed = {cfg['models'].get('confirmed')}")
    subprocess.run(["data-designer", "config", "list"], check=False)
    if test_calls:
        _test_calls(cfg)
    if not ok:
        raise SystemExit(1)


def _test_calls(cfg: dict[str, Any]) -> None:
    import httpx

    for role in ("teacher", "cheap", "judge"):
        m = cfg["models"][role]
        body = {"model": m["slug"], "messages": [{"role": "user", "content": "Reply with the word OK."}],
                "max_tokens": 256, **(m.get("extra_body") or {})}
        if role == "teacher":  # also confirm native tool calling is routed
            body["tools"] = [{"type": "function", "function": {
                "name": "get_time", "description": "Get the current time in a timezone.",
                "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}]
            body["messages"] = [{"role": "user", "content": "What time is it in Tokyo?"}]
        r = httpx.post(f"{cfg['openrouter']['endpoint']}/chat/completions", json=body, timeout=m["timeout"],
                       headers={"Authorization": f"Bearer {api_key(cfg)}"})
        if r.status_code != 200:
            print(f"{role}: HTTP {r.status_code} {r.text[:300]}")
            continue
        j = r.json()
        msg = j["choices"][0]["message"]
        print(f"{role}: provider={j.get('provider')} tool_calls={bool(msg.get('tool_calls'))} "
              f"usage={j.get('usage')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="verify key, slugs, context lengths, prices (Checkpoint 1)")
    c.add_argument("--test-calls", action="store_true", help="one tiny paid call per model")

    for name in ("preview", "run"):
        s = sub.add_parser(name)
        s.add_argument("--n", type=int, default=None)
        s.add_argument("--run-name", default=None)
        s.add_argument("--offline", action="store_true",
                       help="mock server returns stubs, no mock-model calls (teacher/user columns still bill)")
        s.add_argument("--tools", type=Path, default=None, help="override tool library path")
        s.add_argument("--split", type=Path, default=None, help="override domain split path")
    r = sub.choices["run"]
    r.add_argument("--full", action="store_true", help="full run under budget.full_run_usd (after Checkpoint 2)")
    r.add_argument("--yes", action="store_true", help="confirm launching a large paid run")

    rep = sub.add_parser("report", help="re-print stats, cost and samples for a run")
    rep.add_argument("run_name")

    a = p.parse_args()
    cfg = load_config(a.config)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    if a.cmd == "check":
        cmd_check(cfg, a.test_calls)
        return
    if a.cmd == "report":
        run_dir = resolve(cfg["paths"]["raw"]) / a.run_name
        report(cfg, run_dir)
        show_samples(run_dir)
        return

    require_confirmed(cfg)
    api_key(cfg)
    if a.cmd == "preview":
        n = a.n or cfg["generation"]["preview_records"]
        run, scope, cap = a.run_name or "preview", "pilot", cfg["budget"]["pilot_usd"]
    elif a.full:
        cap = cfg["budget"]["full_run_usd"]
        if cap is None:
            raise SystemExit("budget.full_run_usd is not set. Complete Checkpoint 2 and set it first.")
        if not a.yes:
            raise SystemExit("Full run is a large paid job: re-run with --yes to confirm.")
        n, run, scope = a.n or cfg["generation"]["target_generated"], a.run_name or "full", "full"
    else:
        if not a.n:
            raise SystemExit("--n is required for a pilot run.")
        n, run, scope, cap = a.n, a.run_name or "pilot", "pilot", cfg["budget"]["pilot_usd"]

    run_dir = run_generation(cfg, run=run, n=n, cap_usd=cap, cap_scope=scope, offline=a.offline,
                             tools_path=a.tools, split_path=a.split)
    report(cfg, run_dir)
    show_samples(run_dir)
    if a.cmd == "preview":
        print("\nPreview done. Stop here for manual review (Checkpoint 2) before scaling.")


if __name__ == "__main__":
    main()
