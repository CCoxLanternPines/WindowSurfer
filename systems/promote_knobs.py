from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def promote_knobs(run_id: str, tag: str) -> None:
    base = Path(f"data/regimes/{run_id}/tuning/{tag}")
    if not base.exists():
        raise SystemExit(f"[PROMOTE] No tuning results found for run-id {run_id} tag {tag}")

    knobs: Dict[str, Dict] = {}
    for r_dir in sorted(base.glob("R*")):
        best_path = r_dir / "best.json"
        if not best_path.exists():
            continue
        with best_path.open() as fh:
            data = json.load(fh)
        params = data.get("params", data)
        pnl_dd = data.get("pnl_dd", "?")
        trades = data.get("trades", "?")
        regime_key = r_dir.name
        knobs[regime_key] = params
        print(
            f"[PROMOTE] Promoted {regime_key} | pnl_dd={pnl_dd} | trades={trades} using values from best.json"
        )

    seed_path = Path("regimes/seed_knobs.json")
    seed_path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if seed_path.exists():
        with seed_path.open() as fh:
            data = json.load(fh)
    data[tag] = knobs

    with seed_path.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
