import json, re
from functools import lru_cache
from math import isfinite
from typing import Any, Dict, List, Optional, Union

def money(n: float) -> float:
    # bankers-safe rounding to 2dp
    return round((n + 1e-12) * 100) / 100

def _maybe_json(value):
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        try:
            return json.loads(v)
        except Exception:
            return value
    return value

def _coerce_cart_list(v):
    v = _maybe_json(v)
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        out = []
        for s in v:
            try:
                out.append(json.loads(s))
            except Exception:
                pass
        v = out
    out = []
    if isinstance(v, list):
        for it in v:
            if not isinstance(it, dict):
                continue
            if "qty" not in it and "quantity" in it:
                it["qty"] = it.pop("quantity")
            it.setdefault("modifiers", [])
            it.setdefault("combo_opt_in", False)
            it.setdefault("attributes", None)
            out.append(it)
    return out

# utterance → menu hints
TENANT_HINT_RE = re.compile(r"(?:for|from|at|in)\s+([a-z0-9\- _]{2,40})", re.I)
ME_KEYS = {"middle eastern", "middle-eastern", "arabic", "ny", "new york", "new-york"}
US_KEYS = {"american", "usa", "u.s.", "philly", "steak", "burger"}
