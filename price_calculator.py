from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from math import isfinite
import json

app = FastAPI()


# ---------- Utility ----------
def money(n: float) -> float:
    """Round to 2 decimals safely"""
    return round((n + 1e-12) * 100) / 100


# ---------- Data Models ----------
class Mod(BaseModel):
    modifier_id: str
    qty: int = 1
    unit_price: Optional[float] = None


class Line(BaseModel):
    item_id: str
    qty: int
    variant_id: Optional[str] = None
    modifiers: List[Mod] = Field(default_factory=list)
    combo_opt_in: bool = False
    attributes: Optional[Dict[str, Any]] = None


class Payload(BaseModel):
    menu: Union[str, Dict[str, Any]]
    cart: Union[str, List[Line]]
    tax_rate: Union[str, float] = 0.06
    utterances_last_2_turns: Optional[str] = ""

    @validator("menu", pre=True)
    def _coerce_menu(cls, v):
        if v is None:
            return {}
        if isinstance(v, (dict, list)):
            return v
        if isinstance(v, str):
            # agar dynamic var placeholder ho to empty dict
            if v.strip().startswith("knowledge_base"):
                return {}
            try:
                return json.loads(v)
            except Exception:
                return {}
        return {}

    @validator("cart", pre=True)
    def _coerce_cart(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # kuch tools list of JSON strings bhej dete hain -> try to parse whole string as JSON array,
            # agar na ho to split by lines/commas and parse each
            try:
                data = json.loads(v)
                return data
            except Exception:
                try:
                    parts = [p.strip() for p in v.splitlines() if p.strip()] or [p.strip() for p in v.split(",") if p.strip()]
                    parsed = []
                    for p in parts:
                        parsed.append(json.loads(p))
                    return parsed
                except Exception:
                    return []
        return []

    @validator("tax_rate", pre=True)
    def _coerce_tax(cls, v):
        if v in (None, ""):
            return 0.06
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                return 0.06
        return 0.06

# ---------- Core Logic ----------
def index_menu(menu):
    item_idx, mod_idx, cat_mods = {}, {}, {}
    for m in menu.get("modifier_catalog", []):
        mod_idx[m["id"]] = m.get("price", 0)

    for cat in menu.get("categories", []):
        cms = set(cat.get("category_modifiers", []))
        for x in cat.get("extras", []):
            mod_idx[x["id"]] = x["price"]
            cms.add(x["id"])
        cat_mods[cat["id"]] = cms

        for it in cat.get("items", []):
            if "variants" in it and isinstance(it["variants"], list):
                item_idx[it["id"]] = {
                    "type": "variants",
                    "variants": {v["id"]: v["price"] for v in it["variants"]},
                    "category_id": cat["id"]
                }
            else:
                item_idx[it["id"]] = {
                    "type": "flat",
                    "price": it.get("price", 0),
                    "category_id": cat["id"]
                }

    pack = next((p for p in menu.get("pricing_policies", [])
                 if p.get("id") == "pack_optimizer"), None)
    pack_policy = None
    if pack:
        ps = {x["size"]: x["unit_price"] for x in pack.get("packs", [])}
        pack_policy = {
            "applies": set(pack.get("applies_to_item_ids", [])),
            "size": 4,
            "pack": ps.get(4, 9.99),
            "single": ps.get(1, 2.79)
        }

    combo = None
    cp = menu.get("combo_policy")
    if cp:
        combo = {
            "applies": set(cp.get("applies_to_category_ids", [])),
            "excluded": set(cp.get("excluded_category_ids", [])),
            "price": cp.get("combo_extra", {}).get("price", 3.5)
        }

    return item_idx, mod_idx, cat_mods, pack_policy, combo


def resolve_unit(idx, item_id, variant_id):
    info = idx.get(item_id)
    if not info:
        return 0, None
    if info["type"] == "flat":
        return info.get("price", 0), info["category_id"]
    v = variant_id or "base"
    unit = info["variants"].get(v, info["variants"].get("base", 0))
    return unit, info["category_id"]


def sum_mods(mod_idx, line: Line, qty: int) -> float:
    s = 0.0
    for m in line.modifiers:
        unit = (
            m.unit_price
            if (m.unit_price is not None and isfinite(m.unit_price))
            else mod_idx.get(m.modifier_id, 0)
        )
        s += unit * max(1, m.qty or 1)
    return money(s * qty)


def no_deals(u: str) -> bool:
    u = (u or "").lower()
    return "singles only" in u or "no deals" in u or "no bundles" in u


# ---------- Main Endpoint ----------
@app.post("/calc-alsham")
def calc(payload: Payload):
    menu, cart, tax = payload.menu, payload.cart, payload.tax_rate
    item_idx, mod_idx, cat_mods, pack_policy, combo = index_menu(menu)

    lines = []
    pool = []
    block = no_deals(payload.utterances_last_2_turns)

    for ln in cart:
        qty = max(1, ln.qty)
        unit, cat_id = resolve_unit(item_idx, ln.item_id, ln.variant_id)
        mods_total = sum_mods(mod_idx, ln, qty)
        combo_total = 0.0

        if ln.combo_opt_in and combo:
            if cat_id in combo["applies"] and cat_id not in combo["excluded"]:
                combo_total = money(combo["price"] * qty)

        base_only = money(unit * qty)
        line_total = money(base_only + mods_total + combo_total)
        ref = len(lines)

        lines.append({
            "desc": ln.item_id,
            "qty": qty,
            "unit": unit,
            "modifiers_total": mods_total,
            "combo_total": combo_total,
            "line_total": line_total,
            "adjusted": False
        })

        if (not block) and pack_policy and ln.item_id in pack_policy["applies"]:
            pieces_per = 1
            if ln.variant_id == "four":
                pieces_per = 4
            pool.append({"ref": ref, "pieces": qty * pieces_per, "base": base_only})

    optimizer = None
    if (not block) and pack_policy and pool:
        total_pieces = sum(p["pieces"] for p in pool)
        packs = total_pieces // pack_policy["size"]
        singles = total_pieces % pack_policy["size"]

        for p in pool:
            if p["pieces"] > 0:
                lines[p["ref"]]["line_total"] = money(lines[p["ref"]]["line_total"] - p["base"])
                lines[p["ref"]]["adjusted"] = True

        if packs:
            lines.append({
                "desc": "Appetizer Bundle (4 pcs)",
                "qty": packs,
                "unit": pack_policy["pack"],
                "modifiers_total": 0,
                "combo_total": 0,
                "line_total": money(packs * pack_policy["pack"]),
                "optimizer": True
            })
        if singles:
            lines.append({
                "desc": "Appetizer Single (1 pc)",
                "qty": singles,
                "unit": pack_policy["single"],
                "modifiers_total": 0,
                "combo_total": 0,
                "line_total": money(singles * pack_policy["single"]),
                "optimizer": True
            })
        optimizer = {
            "packs": int(packs),
            "singles": int(singles),
            "policy": "min_price_exact_quantity (mixed packs allowed)"
        }

    subtotal = money(sum(l["line_total"] for l in lines))
    tax_amt = money(subtotal * (tax or 0.0))
    total = money(subtotal + tax_amt)

    return {
        "currency": menu.get("meta", {}).get("currency", "USD"),
        "lines": lines,
        "optimizer": optimizer,
        "subtotal": subtotal,
        "tax": tax_amt,
        "total": total
    }
