from typing import Dict, Any, List, Optional

from app.services.menu_loader import load_menu_by_slug, load_default_menu
from app.config import TAX_RATE
from app.utils.common import money
from app.services.menu_loader import resolve_menu_for_line
from app.services.menu_index import (
    index_menu, resolve_unit, sum_mods, category_map
)
from app.utils.common import expand_modifiers_detail
#calculation of the pri
COMBO_ADDON_PRICE = 3.50
COMBO_ELIGIBLE_CATEGORY_IDS = {"long-roll","arabic-sandwiches","maroosh","hoagies","philly-steak","entrees", "entree"}
COMBO_FRIES_IDS = {"side-french-fries"}
COMBO_SODA_IDS  = {"beverage-can-soda", "can-soda", "bev-can-soda"}

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
def apply_combo_adjustments(priced_lines: list) -> None:
    main_idx = None
    for i, pl in enumerate(priced_lines):
        src = pl.get("src") or {}
        combo_opt_in = bool(src.get("combo_opt_in", False))
        cat_id = (src.get("category_id") or pl.get("category_id") or "").lower()
        if combo_opt_in and (cat_id in COMBO_ELIGIBLE_CATEGORY_IDS):
            main_idx = i
            break

    if main_idx is None:
        return

    pl = priced_lines[main_idx]
    pl["combo_total"] = round(pl.get("combo_total", 0.0) + COMBO_ADDON_PRICE, 2)
    pl["line_total"]  = round(pl.get("line_total", 0.0) + COMBO_ADDON_PRICE, 2)
    pl["combo_applied"] = True

    # cover fries + soda IF separate lines exist
    def _cover_one_unit(target):
        qty = int(target.get("qty", 1))
        unit = float(target.get("unit", 0.0))
        if qty >= 1 and unit > 0:
            target["line_total"] = max(0.0, round(target.get("line_total", 0.0) - unit, 2))
            target["combo_covered"] = True

    for t in priced_lines:
        if t.get("desc") in COMBO_FRIES_IDS:
            _cover_one_unit(t); break
    for t in priced_lines:
        if t.get("desc") in COMBO_SODA_IDS:
            _cover_one_unit(t); break



def resolve_menu_for_line(ln: "Line", menu_bundle: dict | None = None) -> dict:
    """Return the correct menu doc for this line based on ln.menu_hint."""
    if menu_bundle:
        hint = (getattr(ln, "menu_hint", None) or "").lower().strip()
        if hint in ("american", "middle-eastern"):
            menu_doc = menu_bundle.get(hint) or {}
            if menu_doc:  # only return if non-empty
                return menu_doc
    # fallback legacy (as a safety)
    # NOTE: uncomment if you want hard fallback to loader
    # try:
    #     return load_menu_by_slug(hint or "middle-eastern")
    # except Exception:
    return {}  # safe empty; caller must guard


def calc_core(
    payload_cart: List,
    payload_utterance: str = "",
    include_tax: bool = True,
    *,
    menu_bundle: dict | None = None,   # ✅ NEW: pass the bundle in
) -> Dict[str, Any]:
    global_utt = payload_utterance or ""
    lines: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []
    optimizer = None
    currency = "USD"

    for ln in payload_cart or []:
        # ✅ single, correct resolve
        menu_for_line = resolve_menu_for_line(ln, menu_bundle=menu_bundle)

        # 🔁 fallback if bundle didn't contain that menu
        if not menu_for_line:
            hint = (getattr(ln, "menu_hint", None) or "middle-eastern").lower()
            try:
                menu_for_line = load_menu_by_slug(hint)
            except Exception:
                # as a last resort, skip the line to avoid crashes
                continue

        # --- build indexes safely
        (item_idx, mod_idx, cat_mods, item_mod_whitelist,
         pack_policies_by_item, menu_combo, category_combo_price) = index_menu(menu_for_line or {})

        # currency per-menu if present
        currency = (menu_for_line.get("meta", {}) or {}).get("currency") or menu_for_line.get("currency") or currency

        cat_name_map = category_map(menu_for_line or {})

        # --- resolve unit/variant
        qty = max(1, int(getattr(ln, "qty", 1) or 1))
        unit, cat_id, cat_name = resolve_unit(item_idx, ln.item_id, ln.variant_id, cat_name_map)
        if unit is None:
            # item not found in this menu; skip safely
            continue

        # --- whitelist modifiers (if defined at item-level)
        if item_mod_whitelist.get(ln.item_id):
            allowed = set(item_mod_whitelist[ln.item_id])
            ln.modifiers = [m for m in (ln.modifiers or []) if m.modifier_id in allowed]

        # --- modifiers price
        mods_total  = sum_mods(mod_idx, ln, qty)
        mods_detail = expand_modifiers_detail(mod_idx, ln)

        # --- combo pricing
        combo_total = 0.0
        if bool(getattr(ln, "combo_opt_in", False)):
            if menu_combo:
                if cat_id in menu_combo["applies"] and cat_id not in menu_combo["excluded"]:
                    combo_price = float(menu_combo.get("price") or 0.0)
                    combo_total = money(combo_price * qty)
            else:
                pc = float(category_combo_price.get(cat_id or "", 0.0))
                if pc:
                    combo_total = money(pc * qty)

        base_only  = money(unit * qty)
        line_total = money(base_only + mods_total + combo_total)

        ref = len(lines)
        lines.append({
            "desc": ln.item_id,  # (optional) you can map to item['name'] if available
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
            "src": {
                "combo_opt_in": bool(getattr(ln, "combo_opt_in", False)),
                "category_id": (cat_id or "").lower(),
                "item_id": str(ln.item_id),
            },
        })

        # --- pack optimizer pool
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

    # --- apply pack optimization
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

            # remove base portions we’re replacing
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

    # (agar tumhari combo adjust hook hai to run karo)
    apply_combo_adjustments(lines)  # keep if implemented

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


# #
# # def resolve_menu_for_line(ln: "Line", menu_bundle: dict | None = None) -> dict:
# #     """Return the correct menu doc for this line based on ln.menu_hint."""
# #     if menu_bundle:
# #         hint = (getattr(ln, "menu_hint", None) or "").lower()
# #         if hint in ("american", "middle-eastern"):
# #             return menu_bundle.get(hint) or {}
# #     # fallback legacy
# #     # return load_menu_by_slug(hint or "middle-eastern")
# #     return {}
#
# def calc_core(payload_cart: List, payload_utterance: str = "", include_tax: bool = True) -> Dict[str, Any]:
#     global_utt = payload_utterance or ""
#     lines: List[Dict[str, Any]] = []
#     pool: List[Dict[str, Any]] = []
#     optimizer = None
#     currency = "USD"
#
#     for ln in payload_cart or []:
#
#         # menu_doc = resolve_menu_for_line(ln, menu_bundle=menu_bundle)
#
#         qty = max(1, ln.qty)
#         menu_for_line = resolve_menu_for_line(ln, payload_utterance)
#         (item_idx, mod_idx, cat_mods, item_mod_whitelist,
#          pack_policies_by_item, menu_combo, category_combo_price) = index_menu(menu_for_line or {})
#         currency = (menu_for_line.get("meta", {}) or {}).get("currency") or menu_for_line.get("currency") or currency
#         cat_name_map = category_map(menu_for_line or {})
#
#         unit, cat_id, cat_name = resolve_unit(item_idx, ln.item_id, ln.variant_id, cat_name_map)
#
#         if item_mod_whitelist.get(ln.item_id):
#             ln.modifiers = [m for m in ln.modifiers if m.modifier_id in item_mod_whitelist[ln.item_id]]
#
#         # modifiers
#         mods_total  = sum_mods(mod_idx, ln, qty)
#         mods_detail = expand_modifiers_detail(mod_idx, ln)
#
#         # combo
#         combo_total = 0.0
#         if ln.combo_opt_in:
#             if menu_combo:
#                 if cat_id in menu_combo["applies"] and cat_id not in menu_combo["excluded"]:
#                     combo_total = money((menu_combo.get("price") or 0) * qty)
#             else:
#                 pc = category_combo_price.get(cat_id or "", 0)
#                 if pc:
#                     combo_total = money(pc * qty)
#
#         base_only  = money(unit * qty)
#         line_total = money(base_only + mods_total + combo_total)
#
#         ref = len(lines)
#         lines.append({
#             "desc": ln.item_id,
#             "qty": qty,
#             "unit": unit,
#             "modifiers_total": mods_total,
#             "modifiers_detail": mods_detail,
#             "combo_total": combo_total,
#             "line_total": line_total,
#             "adjusted": False,
#             "category_id": cat_id,
#             "category": cat_name,
#             "variant_id": (ln.variant_id or "base"),
#             "attributes": (getattr(ln, "attributes", None) or {}),
#             "menu_hint": getattr(ln, "menu_hint", None),
#             "utterance": (getattr(ln, "utterance", None) or payload_utterance or global_utt or ""),
#             "src": {
#                 "combo_opt_in": bool(getattr(ln, "combo_opt_in", False)),
#                 "category_id": (cat_id or "").lower(),
#                 "item_id": str(ln.item_id),
#             },
#
#         })
#
#         # pack optimizer pool
#         this_u = getattr(ln, "utterance", None) or payload_utterance
#         pack_policy = pack_policies_by_item.get(ln.item_id)
#         if pack_policy and (not _no_deals(this_u, pack_policy)):
#             pieces_per = 1
#             vid = (ln.variant_id or "").lower()
#             if vid in ("five", "5", "pack5"):
#                 pieces_per = 5
#             elif vid in ("four", "4", "pack4"):
#                 pieces_per = 4
#
#             pool.append({
#                 "ref": ref,
#                 "item_id": ln.item_id,
#                 "pieces": qty * pieces_per,
#                 "base": base_only,
#                 "policy": pack_policy
#             })
#
#     # apply pack optimization
#     if pool:
#         groups = {}
#         for p in pool:
#             pol = p["policy"]
#             if pol.get("allow_mixed_packs", True):
#                 key = ("mixed", pol["size"], pol["pack"], pol["single"])
#             else:
#                 key = (p.get("item_id") or lines[p["ref"]]["desc"], pol["size"], pol["pack"], pol["single"])
#             groups.setdefault(key, []).append(p)
#
#         optimizer = {"groups": []}
#         for key, same in groups.items():
#             pol = same[0]["policy"]
#             total_pieces = sum(x["pieces"] for x in same)
#             packs   = total_pieces // pol["size"]
#             singles = total_pieces %  pol["size"]
#
#             # remove base price portions we’re replacing
#             for x in same:
#                 if x["pieces"] > 0:
#                     lines[x["ref"]]["line_total"] = money(lines[x["ref"]]["line_total"] - x["base"])
#                     lines[x["ref"]]["adjusted"] = True
#
#             bundle_label = f"Appetizer Bundle ({pol['size']} pcs)"
#             if packs:
#                 lines.append({
#                     "desc": bundle_label,
#                     "qty": int(packs),
#                     "unit": pol["pack"],
#                     "modifiers_total": 0.0,
#                     "combo_total": 0.0,
#                     "line_total": money(packs * pol["pack"]),
#                     "optimizer": True
#                 })
#             if singles:
#                 lines.append({
#                     "desc": "Appetizer Single (1 pc)",
#                     "qty": int(singles),
#                     "unit": pol["single"],
#                     "modifiers_total": 0.0,
#                     "combo_total": 0.0,
#                     "line_total": money(singles * pol["single"]),
#                     "optimizer": True
#                 })
#
#             optimizer["groups"].append({
#                 "allow_mixed": pol.get("allow_mixed_packs", True),
#                 "size": pol["size"],
#                 "packs": int(packs),
#                 "singles": int(singles)
#             })
#     apply_combo_adjustments(lines)  # << add this line
#
#     subtotal = money(sum(l["line_total"] for l in lines))
#     if include_tax:
#         tax_amt = money(subtotal * float(TAX_RATE))
#         total   = money(subtotal + tax_amt)
#     else:
#         tax_amt = 0.0
#         total   = subtotal
#
#     return {
#         "currency": currency,
#         "lines": lines,
#         "optimizer": optimizer,
#         "subtotal": subtotal,
#         "tax_rate": float(TAX_RATE),
#         "tax": tax_amt,
#         "total": total,
#         "include_tax": include_tax,
#         "utterance": payload_utterance,
#
#     }
