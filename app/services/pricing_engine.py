from typing import Dict, Any, List, Optional
from app.config import TAX_RATE
from app.utils.common import money
from app.services.menu_loader import resolve_menu_for_line
from app.services.menu_index import (
    index_menu, resolve_unit, sum_mods, category_map
)

def _no_deals(u: str, pack_policy: Optional[dict]) -> bool:
    u = (u or "").lower()
    if not pack_policy:
        return False
    block_regexes = (pack_policy.get("respect_user_constraint") or {}).get("block_optimizer_if_regex", [])
    import re
    for pat in block_regexes:
        try:
            if re.search(pat, u):
                return True
        except Exception:
            continue
    return ("singles only" in u) or ("no deals" in u) or ("no bundles" in u)

def calc_core(payload_cart: List, payload_utterance: str = "", include_tax: bool = True) -> Dict[str, Any]:
    lines: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []
    optimizer = None
    currency = "USD"

    for ln in payload_cart or []:
        qty = max(1, ln.qty)
        menu_for_line = resolve_menu_for_line(ln, payload_utterance)
        item_idx, mod_idx, cat_mods, item_mod_whitelist, pack_policy, menu_combo, category_combo_price = index_menu(menu_for_line or {})
        currency = (menu_for_line.get("meta", {}) or {}).get("currency") or menu_for_line.get("currency") or currency
        cat_name_map = category_map(menu_for_line or {})

        unit, cat_id, cat_name = resolve_unit(item_idx, ln.item_id, ln.variant_id, cat_name_map)

        if item_mod_whitelist.get(ln.item_id):
            ln.modifiers = [m for m in ln.modifiers if m.modifier_id in item_mod_whitelist[ln.item_id]]

        # modifiers total (reuse imported sum_mods)
        mods_total = sum_mods(mod_idx, ln, qty)

        combo_total = 0.0
        if ln.combo_opt_in:
            if menu_combo:
                if cat_id in menu_combo["applies"] and cat_id not in menu_combo["excluded"]:
                    combo_total = money((menu_combo.get("price") or 0) * qty)
            else:
                pc = category_combo_price.get(cat_id or "", 0)
                if pc:
                    combo_total = money(pc * qty)

        base_only  = money(unit * qty)
        line_total = money(base_only + mods_total + combo_total)

        ref = len(lines)
        lines.append({
            "desc": ln.item_id,
            "qty": qty,
            "unit": unit,
            "modifiers_total": mods_total,
            "combo_total": combo_total,
            "line_total": line_total,
            "adjusted": False,
            "category_id": cat_id,
            "category": cat_name,
        })

        this_u = ln.utterance or payload_utterance
        if pack_policy and (not _no_deals(this_u, pack_policy)) and ln.item_id in pack_policy["applies"]:
            pieces_per = 1
            if ln.variant_id == "four":
                pieces_per = 4
            elif ln.variant_id == "five":
                pieces_per = 0  # intentional: treat as not pack-eligible
            pool.append({"ref": ref, "pieces": qty * pieces_per, "base": base_only, "policy": pack_policy})

    if pool:
        mp = pool[0]["policy"]
        same = [p for p in pool if p["policy"]["size"] == mp["size"] and p["policy"]["pack"] == mp["pack"] and p["policy"]["single"] == mp["single"]]
        total_pieces = sum(p["pieces"] for p in same)
        packs   = total_pieces // mp["size"]
        singles = total_pieces %  mp["size"]

        for p in same:
            if p["pieces"] > 0:
                lines[p["ref"]]["line_total"] = money(lines[p["ref"]]["line_total"] - p["base"])
                lines[p["ref"]]["adjusted"] = True

        if packs:
            lines.append({
                "desc":"Appetizer Bundle (4 pcs)",
                "qty":int(packs),
                "unit":mp["pack"],
                "modifiers_total":0.0,
                "combo_total":0.0,
                "line_total":money(packs*mp["pack"]),
                "optimizer":True
            })
        if singles:
            lines.append({
                "desc":"Appetizer Single (1 pc)",
                "qty":int(singles),
                "unit":mp["single"],
                "modifiers_total":0.0,
                "combo_total":0.0,
                "line_total":money(singles*mp["single"]),
                "optimizer":True
            })

        optimizer = {"packs": int(packs), "singles": int(singles), "policy": "min_price_exact_quantity (mixed packs allowed)"}

    subtotal = money(sum(l["line_total"] for l in lines))

    if include_tax:
        tax_amt  = money(subtotal * float(TAX_RATE))
        total    = money(subtotal + tax_amt)
    else:
        tax_amt  = 0.0
        total    = subtotal

    return {
        "currency": currency,
        "lines": lines,
        "optimizer": optimizer,
        "subtotal": subtotal,
        "tax_rate": float(TAX_RATE),
        "tax": tax_amt,
        "total": total,
        "include_tax": include_tax,   # helpful for UI/tests
    }
