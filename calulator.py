from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from math import isfinite
import json

app = FastAPI()

def money(n: float) -> float:
    return round((n + 1e-12) * 100) / 100

# ---------- Helpers to coerce ELEVEN variants ----------
def _maybe_json(value):
    """Try to JSON-decode a string; otherwise return as-is."""
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        try:
            return json.loads(v)
        except Exception:
            return value  # leave it; caller may handle
    return value

def _coerce_cart_list(v):
    """
    Accepts:
      - JSON string of list
      - list of dicts
      - list of JSON strings
      - list with wrong keys like 'quantity'
    Normalizes to list[dict] with keys: item_id, qty, variant_id, modifiers, combo_opt_in, attributes
    """
    v = _maybe_json(v)
    if v is None:
        return []

    # If still a string (LLM prompt text), try one more time to force parse
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []

    # If it's a list of JSON-strings, parse each element
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        out = []
        for s in v:
            try:
                out.append(json.loads(s))
            except Exception:
                pass
        v = out

    # Now ensure dict shape + key normalization
    out = []
    if isinstance(v, list):
        for it in v:
            if not isinstance(it, dict):
                continue
            # normalize keys
            if "qty" not in it and "quantity" in it:
                it["qty"] = it.pop("quantity")
            it.setdefault("modifiers", [])
            it.setdefault("combo_opt_in", False)
            it.setdefault("attributes", None)
            out.append(it)
    return out

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
    menu: Any
    cart: Union[str, List[Line], List[Any]]
    tax_rate: float = 0.0
    utterances_last_2_turns: Optional[str] = ""

    @validator("menu", pre=True)
    def v_menu(cls, v):
        # Accept dict directly, JSON string, or dynamic-var placeholders
        v = _maybe_json(v)
        if isinstance(v, str) and v.startswith("knowledge_base."):
            return {}  # placeholder, treat as empty
        if v is None:
            return {}
        if not isinstance(v, dict):
            # If menu is invalid, keep empty to avoid 422; your logic can still run
            return {}
        return v

    @validator("cart", pre=True)
    def v_cart(cls, v):
        # Accept anything and normalize to list[dict]; Pydantic will cast to List[Line]
        if isinstance(v, str) and "cart_lines_with" in v:
            return []
        return _coerce_cart_list(v)

# ---------- Pricing logic (unchanged) ----------
def index_menu(menu):
    item_idx, mod_idx, cat_mods = {}, {}, {}
    for m in menu.get("modifier_catalog", []):
        mod_idx[m["id"]] = m.get("price", 0)
    for cat in menu.get("categories", []):
        cms = set(cat.get("category_modifiers", []))
        for x in cat.get("extras", []):
            mod_idx[x["id"]] = x["price"]; cms.add(x["id"])
        cat_mods[cat["id"]] = cms
        for it in cat.get("items", []):
            if "variants" in it and isinstance(it["variants"], list):
                item_idx[it["id"]] = {"type":"variants", "variants":{v["id"]:v["price"] for v in it["variants"]}, "category_id":cat["id"]}
            else:
                item_idx[it["id"]] = {"type":"flat", "price":it.get("price",0), "category_id":cat["id"]}
    pack = next((p for p in menu.get("pricing_policies",[]) if p.get("id")=="pack_optimizer"), None)
    pack_policy = None
    if pack:
        ps = {x["size"]:x["unit_price"] for x in pack.get("packs",[])}
        pack_policy = {"applies": set(pack.get("applies_to_item_ids",[])), "size":4, "pack": ps.get(4,9.99), "single": ps.get(1,2.79)}
    combo = None
    cp = menu.get("combo_policy")
    if cp:
        combo = {"applies": set(cp.get("applies_to_category_ids",[])), "excluded": set(cp.get("excluded_category_ids",[])), "price": cp.get("combo_extra",{}).get("price",3.5)}
    return item_idx, mod_idx, cat_mods, pack_policy, combo

def resolve_unit(idx, item_id, variant_id):
    info = idx.get(item_id)
    if not info: return 0, None
    if info["type"]=="flat": return info.get("price",0), info["category_id"]
    v = variant_id or "base"
    unit = info["variants"].get(v, info["variants"].get("base",0))
    return unit, info["category_id"]

def sum_mods(mod_idx, line: Line, qty: int)->float:
    s=0.0
    for m in line.modifiers:
        unit = m.unit_price if (m.unit_price is not None and isfinite(m.unit_price)) else mod_idx.get(m.modifier_id,0)
        s += unit * max(1, m.qty or 1)
    return money(s * qty)

def no_deals(u: str)->bool:
    u = (u or "").lower()
    return ("singles only" in u) or ("no deals" in u) or ("no bundles" in u)

@app.post("/calc-alsham")
def calc(payload: Payload):
    menu, cart, tax = payload.menu, payload.cart, payload.tax_rate
    item_idx, mod_idx, cat_mods, pack_policy, combo = index_menu(menu or {})

    lines = []
    pool = []
    block = no_deals(payload.utterances_last_2_turns)

    for ln in cart or []:
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
        lines.append({"desc": ln.item_id, "qty": qty, "unit": unit, "modifiers_total": mods_total, "combo_total": combo_total, "line_total": line_total, "adjusted": False})

        if (not block) and pack_policy and ln.item_id in pack_policy["applies"]:
            pieces_per = 1
            if ln.variant_id == "four": pieces_per = 4
            elif ln.variant_id == "five": pieces_per = 0
            pool.append({"ref": ref, "pieces": qty * pieces_per, "base": base_only})

    optimizer=None
    if (not block) and pack_policy and pool:
        total_pieces = sum(p["pieces"] for p in pool)
        packs = total_pieces // pack_policy["size"]
        singles = total_pieces %  pack_policy["size"]
        for p in pool:
            if p["pieces"]>0:
                lines[p["ref"]]["line_total"] = money(lines[p["ref"]]["line_total"] - p["base"])
                lines[p["ref"]]["adjusted"] = True
        if packs:
            lines.append({"desc":"Appetizer Bundle (4 pcs)", "qty":packs, "unit":pack_policy["pack"], "modifiers_total":0, "combo_total":0, "line_total": money(packs*pack_policy["pack"]), "optimizer":True})
        if singles:
            lines.append({"desc":"Appetizer Single (1 pc)", "qty":singles, "unit":pack_policy["single"], "modifiers_total":0, "combo_total":0, "line_total": money(singles*pack_policy["single"]), "optimizer":True})
        optimizer = {"packs": int(packs), "singles": int(singles), "policy": "min_price_exact_quantity (mixed packs allowed)"}

    subtotal = money(sum(l["line_total"] for l in lines))
    tax_amt = money(subtotal * (tax or 0.0))
    total = money(subtotal + tax_amt)

    return {"currency": (menu or {}).get("meta",{}).get("currency","USD"), "lines": lines, "optimizer": optimizer, "subtotal": subtotal, "tax": tax_amt, "total": total}
