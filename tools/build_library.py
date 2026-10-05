"""Phase 1 steps 3 and 5: merge tool sources into the library the generator reads.

Inputs:  tools/library/starter.jsonl         (hand-written tools, edited directly)
         tools/library/real/*.jsonl          (collect_real_tools.py)
Outputs: tools/library/tools.jsonl, tools/library/split.json

Every schema is re-validated, exact duplicates (same name + same schema) are dropped, and the
domain split comes from `library.heldout_domains` in config.yaml. A domain is either train or
held-out, never both; train tools whose name collides with a held-out tool are dropped later
by the generator's leak guard.

    python tools/build_library.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from jsonschema import validators

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config, resolve  # noqa: E402

LIB = ROOT / "tools" / "library"


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    cfg = load_config()
    sources = [LIB / "starter.jsonl", *sorted((LIB / "real").glob("*.jsonl"))]
    tools, seen, dropped = [], set(), Counter()
    for src in sources:
        if not src.exists():
            continue
        for t in _read(src):
            try:
                validators.validator_for(t["inputSchema"]).check_schema(t["inputSchema"])
            except Exception:
                dropped["invalid schema"] += 1
                continue
            key = (t["name"], json.dumps(t["inputSchema"], sort_keys=True))
            if key in seen:
                dropped["exact duplicate"] += 1
                continue
            seen.add(key)
            tools.append(t)

    heldout = list(cfg["library"]["heldout_domains"])
    domains = sorted({t["domain"] for t in tools})
    missing = [d for d in heldout if d not in domains]
    split = {"train_domains": [d for d in domains if d not in heldout], "heldout_domains": heldout}

    out = resolve(cfg["paths"]["tool_library"])
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tools), encoding="utf-8")
    resolve(cfg["paths"]["domain_split"]).write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    by_domain = Counter(t["domain"] for t in tools)
    by_source = Counter("real" if t["id"].startswith("real/") else "starter" for t in tools)
    n_held = sum(by_domain[d] for d in heldout)
    print(f"{len(tools)} tools ({dict(by_source)}), {len(domains)} domains; "
          f"train {len(tools) - n_held} / held-out {n_held}; dropped {dict(dropped)}")
    for d, n in by_domain.most_common():
        print(f"  {d:18s} {n:4d}{'  [held-out]' if d in heldout else ''}")
    if missing:
        print(f"[warn] held-out domains with no tools: {missing}")


if __name__ == "__main__":
    main()
