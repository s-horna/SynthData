# SynthData

Synthetic MCP tool-calling conversations for fine-tuning small language models.

## Setup

```bash
uv venv && uv pip install -r pyproject.toml
cp .env.example .env        # add your OPENROUTER_API_KEY
```

All model slugs, budgets and generation knobs are set in `config.yaml`.

## Usage

```bash
# 1. Build the tool library (already committed; only rerun to refresh it)
python tools/collect_real_tools.py      # pull tool schemas from public MCP servers (needs npx/uvx)
python tools/build_library.py           # merge with starter.jsonl -> tools.jsonl + split.json

# 2. Check models, context lengths and prices
python -m generate.generate check

# 3. Generate
python -m generate.generate preview                      # ~50 rows, review them by hand
python -m generate.generate run --n 500 --run-name pilot  # pilot, capped by budget.pilot_usd
python -m generate.generate run --full --yes             # full run, needs budget.full_run_usd set

# Re-print stats, cost and sample rows for a run
python -m generate.generate report pilot
```

Output goes to `data/raw/<run>/shards/*.parquet`. Spend is recorded in `data/raw/spend_ledger.jsonl`. Runs can be resumed: shards that already finished are skipped.

## How the data is generated

1. **Tool library.** Real MCP tool schemas (permissive licenses only) are combined with hand-written ones. Whole domains (`library.heldout_domains`) are held out for evaluation.
2. **Toolset + scenario per row.** Each row samples 3–20 training tools, occasionally 30–60, including look-alike distractors. It is also assigned one scenario: single call, sequential multi-step (lookup → action), parallel calls, no tool fits, or needs clarification.
3. **User request.** The cheap model writes a request for that scenario using a persona, phrasing style, difficulty and date. The request names the backend records it refers to.
4. **Trajectory.** The teacher model solves the request by calling tools on the **mock MCP server** (`mock_server/`). The server validates arguments against each tool's schema, and the cheap model invents realistic results that contain the request's records. Responses are seeded and cached, so reruns give the same results.
5. **Sharding.** Data Designer caches each MCP server's tool list, so each shard starts several mock-server "slots", one per toolset. Each row goes to its slot, and the results are merged into one parquet file per shard.
