from typing import Dict, Any, Optional, Set, Tuple
from app.utils.common import money

def _norm_cat_modifiers(cat) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    mods = cat.get("category_modifiers") or []
    for m in mods:
        if isinstance(m, str):
            out[m] = None
        elif isinstance(m, dict):
            mid = m.get("id")
            if mid:
                out[mid] = m.get("price")
    return out

def category_map(menu: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for c in menu.get("categories", []):
        cid = c.get("id")
        nm = c.get("name") or cid
        if cid:
            out[cid] = nm
    return out

def index_menu(menu):
    item_idx: Dict[str, Any] = {}
    mod_idx: Dict[str, float] = {}
    cat_allowed_mods: Dict[str, Set[str]] = {}
    item_allowed_mod_ids: Dict[str, Set[str]] = {}
    category_combo_price: Dict[str, float] = {}

    for m in menu.get("modifier_catalog", []):
        mid = m.get("id")
        if not mid:
            continue
        mod_idx[mid] = m.get("price", 0)

    for cat in menu.get("categories", []):
        cat_id = cat["id"]
        cat_mods = _norm_cat_modifiers(cat)
        for mid, p in list(cat_mods.items()):
            if p is None:
                cat_mods[mid] = mod_idx.get(mid, 0)
        cat_allowed_mods[cat_id] = set(cat_mods.keys())
        for mid, p in cat_mods.items():
            if p is not None:
                mod_idx[mid] = p

        combo_opt_id = cat.get("combo_option_id")
        if combo_opt_id:
            category_combo_price[cat_id] = mod_idx.get(combo_opt_id, 0)

        for e in cat.get("extras", []):
            mid = e.get("id")
            if not mid: continue
            mod_idx[mid] = e.get("price", mod_idx.get(mid, 0))
            cat_allowed_mods[cat_id].add(mid)

        for it in cat.get("items", []):
            iid = it["id"]
            allowed = set(it.get("options_allowed", []))
            if allowed:
                item_allowed_mod_ids[iid] = allowed
                for mid in allowed:
                    if mid not in mod_idx:
                        mod_idx[mid] = 0
            if "variants" in it and isinstance(it["variants"], list):
                item_idx[iid] = {"type": "variants","variants": {v["id"]: v for v in it["variants"]},"category_id": cat_id}
            else:
                item_idx[iid] = {"type": "flat","price": it.get("price", 0),"category_id": cat_id}

    pack = next((p for p in menu.get("pricing_policies", []) if p.get("id") == "pack_optimizer"), None)
    pack_policy = None
    if pack:
        ps = {x["size"]: x for x in pack.get("packs", [])}
        pack_policy = {
            "applies": set(pack.get("applies_to_item_ids", [])),
            "size": 4,
            "pack": ps.get(4, {}).get("unit_price", 9.99),
            "single": ps.get(1, {}).get("unit_price", 2.79),
            "respect_user_constraint": pack.get("respect_user_constraint", {}),
            "render_lines": pack.get("render_lines", "split_by_pack"),
            "line_format": pack.get("line_format", "{packs_of_4}× (4 pcs) + {singles}× (1 pc)"),
        }

    combo = None
    cp = menu.get("combo_policy")
    if cp:
        combo = {
            "applies": set(cp.get("applies_to_category_ids", [])),
            "excluded": set(cp.get("excluded_category_ids", [])),
            "price": (cp.get("combo_extra") or {}).get("price", 3.5),
            "ask_instead_of_auto_add": cp.get("ask_instead_of_auto_add", False),
        }

    return item_idx, mod_idx, cat_allowed_mods, item_allowed_mod_ids, pack_policy, combo, category_combo_price

def resolve_unit(idx, item_id, variant_id, cat_name_map):
    info = idx.get(item_id)
    if not info:
        return 0.0, None, None
    cat_id = info["category_id"]
    cat_name = cat_name_map.get(cat_id, cat_id)
    if info["type"] == "flat":
        return float(info.get("price", 0) or 0), cat_id, cat_name
    v = variant_id or "base"
    unit_obj = info["variants"].get(v) or info["variants"].get("base")
    if isinstance(unit_obj, dict):
        return float(unit_obj.get("price", 0) or 0), cat_id, cat_name
    return float(unit_obj or 0), cat_id, cat_name

def sum_mods(mod_idx, line, qty: int) -> float:
    s = 0.0
    for m in line.modifiers:
        unit = m.unit_price if (m.unit_price is not None) else mod_idx.get(m.modifier_id, 0)
        s += float(unit) * max(1, m.qty or 1)
    return money(s * qty)
