import json
import warnings


def load_ledgers(path: str = "settings/ledgers.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ledger_cfg(ledger_name: str, all_cfg: dict) -> dict:
    base = dict(all_cfg.get("default", {}))
    override = dict(all_cfg.get("ledgers", {}).get(ledger_name, {}))
    # Shallow merge is fine given current schema
    base.update(override)

    # Backward compatibility for legacy window_alg structure
    wa = base.pop("window_alg", None)
    if wa is not None:
        warnings.warn(
            "'window_alg' is deprecated; use 'window_size' and 'skip_candles'",
            DeprecationWarning,
            stacklevel=2,
        )
        base.setdefault("window_size", wa.get("window"))
        base.setdefault("skip_candles", int(wa.get("skip_candles", 5)))

    # Ensure skip_candles exists with default of 5
    if "skip_candles" not in base:
        base["skip_candles"] = 5
    else:
        base["skip_candles"] = int(base["skip_candles"])

    return base
