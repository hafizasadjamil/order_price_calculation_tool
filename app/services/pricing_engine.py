from typing import Dict, Any, List, Optional
from app.config import TAX_RATE
from app.utils.common import money
from app.services.menu_loader import resolve_menu_for_line
from app.services.menu_index import (
    index_menu, resolve_unit, sum_mods, category_map
)
from app.utils.common import expand_modifiers_detail

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
    global_utt = payload_utterance or ""
    lines: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []
    optimizer = None
    currency = "USD"

    for ln in payload_cart or []:
        qty = max(1, ln.qty)
        menu_for_line = resolve_menu_for_line(ln, payload_utterance)
        (item_idx, mod_idx, cat_mods, item_mod_whitelist,
         pack_policies_by_item, menu_combo, category_combo_price) = index_menu(menu_for_line or {})
        currency = (menu_for_line.get("meta", {}) or {}).get("currency") or menu_for_line.get("currency") or currency
        cat_name_map = category_map(menu_for_line or {})

        unit, cat_id, cat_name = resolve_unit(item_idx, ln.item_id, ln.variant_id, cat_name_map)

        if item_mod_whitelist.get(ln.item_id):
            ln.modifiers = [m for m in ln.modifiers if m.modifier_id in item_mod_whitelist[ln.item_id]]

        # modifiers
        mods_total  = sum_mods(mod_idx, ln, qty)
        mods_detail = expand_modifiers_detail(mod_idx, ln)

        # combo
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
            "modifiers_detail": mods_detail,
            "combo_total": combo_total,
            "line_total": line_total,
            "adjusted": False,
            "category_id": cat_id,
            "category": cat_name,
            "variant_id": (ln.variant_id or "base"),
            "attributes": (getattr(ln, "attributes", None) or {}),
            "menu_hint": getattr(ln, "menu_hint", None),
            "utterance": (getattr(ln, "utterance", None) or payload_utterance or global_utt or ""),

        })

        # pack optimizer pool
        this_u = getattr(ln, "utterance", None) or payload_utterance
        pack_policy = pack_policies_by_item.get(ln.item_id)
        if pack_policy and (not _no_deals(this_u, pack_policy)):
            pieces_per = 1
            vid = (ln.variant_id or "").lower()
            if vid in ("five", "5", "pack5"):
                pieces_per = 5
            elif vid in ("four", "4", "pack4"):
                pieces_per = 4

            pool.append({
                "ref": ref,
                "item_id": ln.item_id,
                "pieces": qty * pieces_per,
                "base": base_only,
                "policy": pack_policy
            })

    # apply pack optimization
    if pool:
        groups = {}
        for p in pool:
            pol = p["policy"]
            if pol.get("allow_mixed_packs", True):
                key = ("mixed", pol["size"], pol["pack"], pol["single"])
            else:
                key = (p.get("item_id") or lines[p["ref"]]["desc"], pol["size"], pol["pack"], pol["single"])
            groups.setdefault(key, []).append(p)

        optimizer = {"groups": []}
        for key, same in groups.items():
            pol = same[0]["policy"]
            total_pieces = sum(x["pieces"] for x in same)
            packs   = total_pieces // pol["size"]
            singles = total_pieces %  pol["size"]

            # remove base price portions we’re replacing
            for x in same:
                if x["pieces"] > 0:
                    lines[x["ref"]]["line_total"] = money(lines[x["ref"]]["line_total"] - x["base"])
                    lines[x["ref"]]["adjusted"] = True

            bundle_label = f"Appetizer Bundle ({pol['size']} pcs)"
            if packs:
                lines.append({
                    "desc": bundle_label,
                    "qty": int(packs),
                    "unit": pol["pack"],
                    "modifiers_total": 0.0,
                    "combo_total": 0.0,
                    "line_total": money(packs * pol["pack"]),
                    "optimizer": True
                })
            if singles:
                lines.append({
                    "desc": "Appetizer Single (1 pc)",
                    "qty": int(singles),
                    "unit": pol["single"],
                    "modifiers_total": 0.0,
                    "combo_total": 0.0,
                    "line_total": money(singles * pol["single"]),
                    "optimizer": True
                })

            optimizer["groups"].append({
                "allow_mixed": pol.get("allow_mixed_packs", True),
                "size": pol["size"],
                "packs": int(packs),
                "singles": int(singles)
            })

    subtotal = money(sum(l["line_total"] for l in lines))
    if include_tax:
        tax_amt = money(subtotal * float(TAX_RATE))
        total   = money(subtotal + tax_amt)
    else:
        tax_amt = 0.0
        total   = subtotal

    return {
        "currency": currency,
        "lines": lines,
        "optimizer": optimizer,
        "subtotal": subtotal,
        "tax_rate": float(TAX_RATE),
        "tax": tax_amt,
        "total": total,
        "include_tax": include_tax,
        "utterance": payload_utterance,

    }
