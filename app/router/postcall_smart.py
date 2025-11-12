# app/router/rest_ingest_v2.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List,Tuple
import os, json, re
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.schemas.models import Line, Mod, Payload
from app.services.pricing_engine import calc_core
from app.services.menu_loader import load_menu_by_slug, load_default_menu

load_dotenv()
router = APIRouter(prefix="/rest", tags=["rest"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EXTRACT_MODEL = os.getenv("EXTRACTOR_MODEL", "gpt-4o-mini")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")


COMBO_MOD_RE = re.compile(r"(combo|fries.*(can|soda)|fries.*drink)", re.I)
NAME_RE = re.compile(r"The user,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
PICKUP_RE = re.compile(r"\b(pick\s*up|pickup|pick-up)\b", re.I)
DELIVERY_RE = re.compile(r"\bdeliver(y)?\b", re.I)
# Tokens for explicit packs
PACK_TOKENS_RE = re.compile(r"\b(pack|deal|combo|pcs|pieces|piece|pc)\b", re.I)
# Per-item piece sizes you support (customize per menu)


def _is_appetizer(iid: str, schema_ctx: dict) -> bool:
    cat = (schema_ctx.get("category_by_item") or {}).get(iid, "")
    return bool(cat and ("appetizer" in cat.lower() or cat.lower().startswith("app-")))

def _normalize_appetizer_packs(transcript: str, lines: list[dict], schema_ctx: dict) -> list[dict]:
    """
    If user said '5 samosas' with no pack token, make it qty=5, variant='single'.
    If user said '5 pcs' or 'pack', keep/flip to a pack variant if it exists, otherwise leave as explicit qty singles.
    """
    t = (transcript or "").lower()
    is_pack_context = bool(PACK_TOKENS_RE.search(t))

    out = []
    for ln in (lines or []):
        iid = ln.get("item_id","")
        if not _is_appetizer(iid, schema_ctx):
            out.append(ln); continue

        qty = int(ln.get("qty", 1) or 1)
        vid = ln.get("variant_id") or "base"
        valid_vs = (schema_ctx.get("variants_by_item") or {}).get(iid) or ["base"]

        if is_pack_context:
            # if a pack variant exists in your catalog and qty=1, keep as pack; else leave as-is
            # (we avoid inventing pack math; pricing engine can still optimize)
            out.append(ln)
            continue

        # No pack token → singles semantics
        if "single" in valid_vs:
            ln["variant_id"] = "single"
        elif "base" in valid_vs:
            ln["variant_id"] = "base"
        # qty stays whatever the user said; pricing can still convert later if it has pack optimizers
        out.append(ln)
    return out

def _flatten_eleven_transcript(payload: Dict[str, Any], include_agent: bool = True) -> str:
    """
    Build a single text transcript that our extractor can read.
    You can choose to include both sides (agent+user) or only user.
    """
    turns = payload.get("transcript") or []
    parts = []
    for t in turns:
        role = t.get("role")
        msg = (t.get("message") or "").strip()
        if not msg:
            continue
        if role == "user":
            parts.append(f"User: {msg}")
        elif include_agent and role == "agent":
            parts.append(f"Agent: {msg}")
    return "\n".join(parts)

def _extract_name(payload: Dict[str, Any]) -> str:
    # Prefer analysis.transcript_summary if present
    summary = (((payload.get("analysis") or {}).get("transcript_summary")) or "")[:500]
    m = NAME_RE.search(summary)
    if m:
        return m.group(1).strip()
    # fallback: try to detect a “name” style in user turns (very light)
    for t in payload.get("transcript") or []:
        if t.get("role") == "user":
            msg = (t.get("message") or "")
            if "name is" in msg.lower():
                # crude split
                return msg.split("is", 1)[-1].strip().split(".")[0]
    return ""

def _detect_order_type(transcript_text: str) -> str:
    if PICKUP_RE.search(transcript_text):
        return "pickup"
    if DELIVERY_RE.search(transcript_text):
        return "delivery"
    return "unspecified"

def _get_caller_phone(payload: Dict[str, Any]) -> str:
    return (((payload.get("metadata") or {}).get("phone_call") or {}).get("external_number")) or ""
def _enforce_explicit_combos(pricing: dict, src_lines: list) -> dict:
    """
    Ensure combos are charged ONLY when the source line explicitly had combo_opt_in=True.
    If the pricing engine upsells combos automatically, remove those combo charges and
    recompute totals safely.
    """
    if not pricing:
        return pricing

    def _attr_key(a: dict) -> tuple:
        items = []
        for k, v in (a or {}).items():
            if k == "_notes":
                items.append((k, tuple(str(x) for x in (v or []))))
            else:
                items.append((k, v if isinstance(v, (str, int, float, bool, type(None))) else str(v)))
        return tuple(sorted(items))

    # Build a lookup of "allowed combo lines" from extractor output
    allow_combo_counts = {}
    for ln in src_lines or []:
        k = (
            getattr(ln, "item_id", None),
            getattr(ln, "variant_id", "base"),
            getattr(ln, "menu_hint", "middle-eastern"),
            _attr_key(getattr(ln, "attributes", {}) or {}),
            tuple(sorted((m.modifier_id, int(m.qty or 1)) for m in (ln.modifiers or []))),
        )
        if getattr(ln, "combo_opt_in", False):
            allow_combo_counts[k] = allow_combo_counts.get(k, 0) + getattr(ln, "qty", 1)

    subtotal = 0.0
    include_tax = bool(pricing.get("include_tax", True))
    tax_rate = float(pricing.get("tax_rate", 0.0) or 0.0)

    for ln in pricing.get("lines", []) or []:
        src = ln.get("src") or {}
        attrs = src.get("attributes") or ln.get("attributes") or {}
        mods = src.get("modifiers") or ln.get("modifiers") or []
        k = (
            (src.get("item_id") or ln.get("item_id")),
            (src.get("variant_id") or ln.get("variant_id") or "base"),
            (src.get("menu_hint") or ln.get("menu_hint") or "middle-eastern"),
            _attr_key(attrs),
            tuple(sorted((m.get("modifier_id"), int(m.get("qty", 1))) for m in (mods or []) if m.get("modifier_id"))),
        )

        qty = int(ln.get("qty", 1) or 1)
        combo_total = float(ln.get("combo_total", 0.0) or 0.0)

        if combo_total > 0.0:
            allowed = allow_combo_counts.get(k, 0)
            if allowed <= 0:
                # This line should NOT have combo — strip it
                ln["combo_total"] = 0.0
                try:
                    ln["line_total"] = float(ln.get("line_total", 0.0) or 0.0) - combo_total
                    if ln["line_total"] < 0:
                        ln["line_total"] = 0.0
                except Exception:
                    pass
            else:
                # Consume allowed combos up to qty
                allow_combo_counts[k] = max(0, allowed - qty)

        try:
            subtotal += float(ln.get("line_total", 0.0) or 0.0)
        except Exception:
            pass

    # Recompute totals
    pricing["subtotal"] = round(subtotal, 2)
    if include_tax:
        tax = round(subtotal * tax_rate, 2)
        pricing["tax"] = tax
        pricing["total"] = round(subtotal + tax, 2)
    else:
        pricing["tax"] = 0.0
        pricing["total"] = round(subtotal, 2)

    return pricing


def _repair_llm_lines(raw_lines: list[dict]) -> list[dict]:
    fixed = []
    for ln in raw_lines or []:
        if not isinstance(ln, dict):
            continue
        d = dict(ln)
        # quantity → qty
        if "quantity" in d and "qty" not in d:
            try:
                d["qty"] = int(d["quantity"])
            except Exception:
                d["qty"] = 1
            d.pop("quantity", None)

        # ids must be strings
        if "item_id" in d:
            d["item_id"] = str(d["item_id"]).strip()

        # normalize modifiers
        mods = []
        for m in d.get("modifiers", []) or []:
            if not isinstance(m, dict):
                continue
            mm = dict(m)
            if "quantity" in mm and "qty" not in mm:
                try:
                    mm["qty"] = int(mm["quantity"])
                except Exception:
                    mm["qty"] = 1
                mm.pop("quantity", None)
            if "modifier_id" in mm:
                mm["modifier_id"] = str(mm["modifier_id"]).strip()
            mods.append(mm)
        d["modifiers"] = mods
        fixed.append(d)
    return fixed

# --- BOTH-MENU SUPPORT ---
def _build_item_context(bundle: dict) -> tuple[dict, dict, dict]:
    """
    Returns:
      item_names_by_id: {iid: "Human Name"}
      aliases_by_item: {iid: ["alias1","alias2",...]}
      category_by_item: {iid: "category_id"}
    """
    item_names_by_id = {}
    aliases_by_item = {}
    category_by_item = {}

    for hint in ("american","middle-eastern"):
        menu = bundle.get(hint) or {}

        # aliases are optional per menu
        alias_map = (menu.get("alias_map") or {}) if isinstance(menu.get("alias_map"), dict) else {}

        for cat in (menu.get("categories") or []):
            cat_id = cat.get("id")
            for it in (cat.get("items") or []):
                iid = it.get("id")
                if not iid:
                    continue
                item_names_by_id[iid] = it.get("name") or iid
                category_by_item[iid] = cat_id or ""
                # pull aliases if defined in alias_map keyed by iid
                aliases_by_item[iid] = list(alias_map.get(iid, []))

    return item_names_by_id, aliases_by_item, category_by_item

def _merge_duplicate_lines(cart: List[Dict]) -> List[Dict]:
    """
    Merge logically-identical cart lines by summing qty.
    Identity key includes:
      - item_id
      - variant_id (default "base")
      - menu_hint (default "middle-eastern")
      - combo_opt_in (bool)
      - attributes (notes-only, order-insensitive)
      - normalized modifiers (sorted by id, aggregated qty)
    """
    def _norm_mods(mods: List[Dict]) -> Tuple[Tuple[str, int], ...]:
        acc = {}
        for m in mods or []:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("modifier_id") or "").strip()
            if not mid:
                continue
            try:
                mqty = int(m.get("qty", 1) or 1)
            except Exception:
                mqty = 1
            if mqty < 1:
                mqty = 1
            acc[mid] = acc.get(mid, 0) + mqty
        # sorted, immutable form for key; also used to rehydrate merged line
        return tuple(sorted(acc.items()))

    def _attr_key(attrs: Dict) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """
        Only consider notes for identity. Coerce to tuple, order-insensitive, deduped.
        """
        if not isinstance(attrs, dict):
            return tuple()
        notes = attrs.get("_notes")
        if notes is None:
            return tuple()
        if isinstance(notes, str):
            notes = [notes]
        # trim whitespace, keep short (≤80 chars), dedupe case-insensitively
        clean, seen = [], set()
        for n in notes or []:
            s = " ".join(str(n).split())[:80]
            k = s.lower()
            if s and k not in seen:
                clean.append(s)
                seen.add(k)
        return (("_notes", tuple(sorted(clean))),) if clean else tuple()

    acc = {}
    order = []  # preserve first-seen order of keys

    for ln in cart or []:
        if not isinstance(ln, dict):
            continue

        iid = str(ln.get("item_id") or "").strip()
        if not iid:
            continue

        vid  = (ln.get("variant_id") or "base")
        hint = (ln.get("menu_hint") or "middle-eastern")
        combo = bool(ln.get("combo_opt_in", False))

        # normalize modifiers for identity and storage
        mods_key = _norm_mods(ln.get("modifiers"))
        attrs_key = _attr_key(ln.get("attributes") or {})

        key = (iid, vid, hint, combo, mods_key, attrs_key)

        # coerce qty
        try:
            qty = max(1, int(ln.get("qty", 1) or 1))
        except Exception:
            qty = 1

        if key not in acc:
            # create a normalized copy
            base = {
                "item_id": iid,
                "variant_id": vid,
                "menu_hint": hint,
                "combo_opt_in": combo,
                "qty": qty,
                # rehydrate modifiers from mods_key
                "modifiers": [{"modifier_id": mid, "qty": mqty} for mid, mqty in mods_key],
                # rehydrate attributes from attrs_key
                "attributes": {"_notes": list(attrs_key[0][1])} if attrs_key else {},
            }
            # carry through any extra (safe) fields if present
            for extra in ("desc", "src"):
                if extra in ln:
                    base[extra] = ln[extra]
            acc[key] = base
            order.append(key)
        else:
            acc[key]["qty"] += qty

    return [acc[k] for k in order]

def _load_menu_bundle(pref_slug: str | None = None) -> dict:
    """
    Returns a bundle:
    { 'american': <menu or {}>, 'middle-eastern': <menu or {}>, 'preferred': 'american'|'middle-eastern' }
    pref_slug is a hint (e.g., from collected['menu_hint']) but both are loaded.
    """
    try:
        me = load_menu_by_slug("middle-eastern")
    except Exception:
        me = load_default_menu()
    try:
        am = load_menu_by_slug("american")
    except Exception:
        am = {}

    preferred = (pref_slug or "middle-eastern")
    if preferred not in {"american", "middle-eastern"}:
        preferred = "middle-eastern"

    return {"american": am or {}, "middle-eastern": me or {}, "preferred": preferred}


def _handoff_from_payload(payload: Dict[str, Any]) -> dict | None:
    # Accept several shapes used by tools/webhooks
    status = (payload.get("status") or "").strip().lower()
    if status in {"transferred", "transfer", "handoff"}:
        return {"requested": True, "target": "store_number"}
    if ((payload.get("analysis") or {}).get("handoff") is True) or \
       ((payload.get("metadata") or {}).get("handoff") is True):
        return {"requested": True, "target": "store_number"}
    return None


# --- intent & status helpers (add near other regexes) ---
# TRANSFER_RE = re.compile(r"\b(transfer|connect|speak to|talk to)\b.*\b(manager|owner|chef|kitchen|human|agent)\b", re.I)
CALLCUT_RE = re.compile(r"\b(call (dropped|cut)|hello\?\s*you there|are you there\??|disconnected)\b", re.I)
INFO_HINTS_RE = re.compile(r"\b(hours?|open|close|address|location|where are you|menu|price|how much|deliver(y)? radius|do you have)\b", re.I)
PAYMENT_ISSUE_RE = re.compile(
    r"\b(online (order|payment)|card (not|isn'?t) working|payment (issue|problem|failed|declined)|refund|charge\s*back|charged\s*twice|app (issue|problem))\b",
    re.I,
)
TRANSFER_RE = re.compile(
    r"\b(transfer|connect|speak|talk|forward|send|pass|put.*through)\b.*\b(manager|owner|chef|kitchen|human|agent|store|restaurant|cashier|front\s*desk|billing|payment)\b",
    re.I,
)
def _detect_transfer(t: str) -> dict | None:
    m = TRANSFER_RE.search(t or "")
    if not m: return None
    target = m.group(2).lower()
    if "manager" in target: target = "manager"
    elif "chef" in target or "kitchen" in target: target = "kitchen"
    elif "owner" in target: target = "owner"
    elif "store" in target or "restaurant" in target: target = "store_number"
    else: target = "human_agent"
    return {"requested": True, "target": target}


def _detect_transfer_both_sides(t_with_agent: str, t_user_only: str) -> dict | None:
    # 1) Any explicit transfer phrase by agent wins immediately
    for line in (t_with_agent or "").splitlines():
        if line.startswith("Agent:") and TRANSFER_RE.search(line):
            m = TRANSFER_RE.search(line)
            target_raw = m.group(2).lower() if m else ""
            if "manager" in target_raw: target = "manager"
            elif "chef" in target_raw or "kitchen" in target_raw: target = "kitchen"
            elif "owner" in target_raw: target = "owner"
            elif "store" in target_raw or "restaurant" in target_raw or "cashier" in target_raw or "front" in target_raw or "billing" in target_raw or "payment" in target_raw:
                target = "store_number"
            else:
                target = "human_agent"
            return {"requested": True, "target": target}

    # 2) User explicitly requests transfer
    if TRANSFER_RE.search(t_user_only or ""):
        m = TRANSFER_RE.search(t_user_only)
        target_raw = m.group(2).lower() if m else ""
        if "manager" in target_raw: target = "manager"
        elif "chef" in target_raw or "kitchen" in target_raw: target = "kitchen"
        elif "owner" in target_raw: target = "owner"
        elif "store" in target_raw or "restaurant" in target_raw or "billing" in target_raw or "payment" in target_raw:
            target = "store_number"
        else:
            target = "human_agent"
        return {"requested": True, "target": target}

    # 3) Payment/online-order issue + any agent promise to transfer (implicit)
    if PAYMENT_ISSUE_RE.search(t_with_agent or ""):
        # Heuristic: if agent says "I'll transfer / connect / forward you"
        agent_offers = re.search(r"Agent:.*\b(transfer|connect|forward|send|put.*through)\b", t_with_agent or "", re.I)
        if agent_offers:
            return {"requested": True, "target": "store_number"}

    return None

def _detect_call_cut(t: str) -> bool:
    return bool(CALLCUT_RE.search(t or ""))

def _detect_info_only(t: str) -> bool:
    return bool(INFO_HINTS_RE.search(t or ""))

def _last_intent_wins(transcript: str, schema_ctx: dict, cart_lines: list[dict], cancel_seen: bool) -> bool:
    """
    Return True if final intent is CANCEL (i.e., 'no order'), False otherwise.
    If cancel appears but a NEW order intent occurs *after* cancel, cancel is ignored.
    """
    if not cancel_seen:
        return False
    t = transcript or ""
    cancel_last = max([m.start() for m in CANCEL_RE.finditer(t)] or [-1])
    if cancel_last < 0:
        return False

    # Build a light item matcher from known ids and names
    ids = schema_ctx.get("allowed_item_ids", [])
    names = []
    for hint in ("american","middle-eastern"):
        menu = (schema_ctx.get("_menus") or {}).get(hint) or {}
        for cat in (menu.get("categories") or []):
            for it in (cat.get("items") or []):
                nm = it.get("name")
                if nm: names.append(re.escape(nm.lower()))
    # any of these after 'cancel' indicates a new order intent
    POST_CANCEL_ORDER_RE = re.compile(r"(" + "|".join([re.escape(i) for i in ids][:200]) + "|" + "|".join(names[:200]) + r")", re.I) if (ids or names) else None
    if POST_CANCEL_ORDER_RE and POST_CANCEL_ORDER_RE.search(t, pos=cancel_last+1):
        return False   # new order after cancel → cancel ignored
    # also if we actually extracted cart_lines, treat as new intent (extraction is over full transcript)
    if cart_lines:
        # If any attributes._notes contain “restart” etc., also ignore cancel
        return False
    return True  # final intent truly cancel

GENERIC_APP_IDS = {"app-single", "appetizer-single", "app-1pc", "app-single-1pc"}  # adjust to your real IDs
from typing import Callable

CategoryPolicy = Callable[[str, List[Dict], dict], List[Dict]]

def _policy_appetizers(transcript: str, lines: List[Dict], schema_ctx: dict) -> List[Dict]:
    # singles-by-default unless explicit pack tokens found in transcript
    lines = _normalize_appetizer_packs(transcript, lines, schema_ctx)
    return lines

def _policy_drinks(transcript: str, lines: List[Dict], schema_ctx: dict) -> List[Dict]:
    # remove drinks unless explicitly ordered (you already enforce in prompt; this is a second guard)
    out = []
    bev_ids = schema_ctx.get("beverage_item_ids") or set()
    said_drink = bool(re.search(r"\b(water|coke|sprite|fanta|snapple|can of soda|soda|drink|juice|tea)\b", transcript.lower()))
    for ln in lines or []:
        if ln.get("item_id") in bev_ids and not said_drink:
            # keep only if combo notes explicitly say a drink choice
            notes = ((ln.get("attributes") or {}).get("_notes") or [])
            if not any("for combo" in str(n).lower() for n in notes):
                continue
        out.append(ln)
    return out

def _policy_combos(transcript: str, lines: List[Dict], schema_ctx: dict) -> List[Dict]:
    return _reconcile_combo(transcript, lines)

def _priceability_gate(lines: List[Dict], schema_ctx: dict) -> Tuple[List[Dict], List[str]]:
    reasons = []
    vmap = schema_ctx.get("variants_by_item") or {}
    out = []
    for ln in lines or []:
        iid = ln.get("item_id")
        vid = ln.get("variant_id") or "base"
        valid = vmap.get(iid) or ["base"]
        if vid not in valid:
            # deterministic repair
            fix = "single" if "single" in valid else ("base" if "base" in valid else valid[0])
            if fix != vid:
                ln["variant_id"] = fix
        # after repair, recheck
        if (ln.get("variant_id") or "base") not in (vmap.get(iid) or []):
            reasons.append(f"Unpriceable line dropped: {iid} [{vid}]")
            continue
        out.append(ln)
    return out, reasons

# map by *category label contains* to a policy func
CATEGORY_POLICIES: list[tuple[re.Pattern, CategoryPolicy]] = [
    (re.compile(r"appetizer", re.I), _policy_appetizers),
    (re.compile(r"drink|beverage|bev", re.I), _policy_drinks),
    # (re.compile(r".*"), _policy_combos),  # combos apply to all categories
]

def _apply_category_policies(transcript: str, lines: List[Dict], schema_ctx: dict) -> List[Dict]:
    """Apply all matching category policies to each line."""
    cat_by_item = schema_ctx.get("category_by_item") or {}
    grouped: Dict[str, List[Dict]] = {}
    for ln in lines or []:
        cid = cat_by_item.get(ln.get("item_id"), "")
        grouped.setdefault(cid, []).append(ln)

    out = []
    for cid, group in grouped.items():
        g = group
        for pat, policy in CATEGORY_POLICIES:
            if pat.search(cid or ""):
                g = policy(transcript, g, schema_ctx)
        out.extend(g)
    return out

def _drop_generic_appetizer_placeholders(lines: list[dict]) -> list[dict]:
    item_ids = {ln.get("item_id") for ln in (lines or [])}
    has_concrete_app = any((iid not in GENERIC_APP_IDS) and iid for iid in item_ids)
    if not has_concrete_app:
        return lines
    return [ln for ln in (lines or []) if ln.get("item_id") not in GENERIC_APP_IDS]

def _index_placeholders(menu_bundle: dict) -> set[str]:
    """
    Detect generic/placeholder items via common heuristics in *any* category.
    If your JSON can mark them (e.g., is_placeholder: true), prefer that.
    """
    placeholders = set()
    keys = ("placeholder", "generic", "any", "single", "build your own", "choose one")
    for hint in ("american", "middle-eastern"):
        menu = menu_bundle.get(hint) or {}
        for cat in (menu.get("categories") or []):
            for it in (cat.get("items") or []):
                iid = it.get("id")
                nm  = (it.get("name") or "").lower()
                if it.get("is_placeholder") is True:
                    placeholders.add(iid); continue
                if any(k in nm for k in keys):
                    placeholders.add(iid)
    return placeholders


def _prefer_specific_over_placeholder(lines: List[Dict], schema_ctx: dict, placeholders: set[str]) -> List[Dict]:
    cat_by_item = schema_ctx.get("category_by_item") or {}
    # group by category_id
    by_cat: Dict[str, List[Dict]] = {}
    for ln in lines or []:
        iid = ln.get("item_id")
        cid = cat_by_item.get(iid, "")
        by_cat.setdefault(cid, []).append(ln)

    out = []
    for cid, group in by_cat.items():
        has_specific = any(g.get("item_id") not in placeholders for g in group)
        if has_specific:
            out.extend([g for g in group if g.get("item_id") not in placeholders])
        else:
            out.extend(group)
    return out


import logging
log = logging.getLogger(__name__)
def _union_schema_from_bundle(bundle: dict) -> dict:
    """
    Build a strict, unioned schema across the american + middle-eastern menus.

    Returns a dict with:
      - allowed_item_ids: [str]
      - allowed_modifier_ids: [str]
      - variants_by_item: {item_id: [variant_id| 'base']}
      - item_menu_hint: {item_id: 'american'|'middle-eastern'}
      - item_names_by_id: {item_id: human-readable name}
      - aliases_by_item: {item_id: [alias, ...]}
      - category_by_item: {item_id: category_id}
      - beverage_item_ids: set([...])             # convenience: ids starting with 'bev-'
      - specials_block_combo: set([...])          # items that must never set combo_opt_in
      - _menus: {'american': {...}, 'middle-eastern': {...}}  # raw menus (unchanged)
    """
    import logging
    log = logging.getLogger(__name__)

    item_ids: list[str] = []
    modifier_ids: list[str] = []
    variants_by_item: dict[str, list[str]] = {}
    item_menu_hint: dict[str, str] = {}

    # NEW: richer context for strict matching
    item_names_by_id: dict[str, str] = {}
    aliases_by_item: dict[str, list[str]] = {}
    category_by_item: dict[str, str] = {}

    beverage_item_ids: set[str] = set()
    specials_block_combo: set[str] = set()

    seen_item_owner: dict[str, str] = {}   # item_id -> first menu key seen
    seen_modifier_owner: dict[str, str] = {}  # modifier_id -> first menu key seen

    for hint_key in ("american", "middle-eastern"):
        menu_doc = bundle.get(hint_key) or {}

        # --- global modifier catalog
        for m in (menu_doc.get("modifier_catalog") or []):
            mid = m.get("id")
            if not mid:
                continue
            if mid in seen_modifier_owner and seen_modifier_owner[mid] != hint_key:
                log.warning(
                    "Duplicate modifier_id across menus detected: %s  (first=%s, now=%s)",
                    mid, seen_modifier_owner[mid], hint_key
                )
            seen_modifier_owner.setdefault(mid, hint_key)
            modifier_ids.append(mid)

        # top-level alias map (optional) — maps item_id -> [aliases...]
        top_alias_map = menu_doc.get("alias_map") or {}

        for cat in (menu_doc.get("categories") or []):
            cat_id = cat.get("id") or ""

            # category-level modifiers (can be str or {id,...})
            for mod in (cat.get("category_modifiers") or []):
                if isinstance(mod, dict) and mod.get("id"):
                    modifier_ids.append(mod["id"])
                elif isinstance(mod, str):
                    modifier_ids.append(mod)

            # extras array (each {id,...})
            for ex in (cat.get("extras") or []):
                mid = ex.get("id")
                if mid:
                    modifier_ids.append(mid)

            # items
            for it in (cat.get("items") or []):
                iid = it.get("id")
                if not iid:
                    continue

                # cross-menu collision detection
                if iid in seen_item_owner and seen_item_owner[iid] != hint_key:
                    log.warning(
                        "Duplicate item_id across menus detected: %s  (first=%s, now=%s)",
                        iid, seen_item_owner[iid], hint_key
                    )
                seen_item_owner.setdefault(iid, hint_key)

                item_ids.append(iid)
                item_menu_hint[iid] = hint_key
                category_by_item[iid] = cat_id
                item_names_by_id[iid] = (it.get("name") or iid)

                # gather aliases: from top-level alias_map (preferred), else empty list
                aliases_by_item[iid] = list(top_alias_map.get(iid, []))

                # variants (default to "base" when none defined)
                if isinstance(it.get("variants"), list) and it["variants"]:
                    v_ids = [v.get("id") for v in it["variants"] if v.get("id")]
                    variants_by_item[iid] = v_ids if v_ids else ["base"]
                else:
                    variants_by_item[iid] = ["base"]

                # convenience flags
                if iid.startswith("bev-"):
                    beverage_item_ids.add(iid)

                # never upsell combos for fixed specials / items that suppress combos
                if it.get("suppress_combo_prompt") is True or it.get("is_combo") is True:
                    specials_block_combo.add(iid)

    # dedupe & sort
    item_ids = sorted(set(item_ids))
    modifier_ids = sorted(set(modifier_ids))

    # build output
    return {
        "allowed_item_ids": item_ids,
        "allowed_modifier_ids": modifier_ids,
        "variants_by_item": variants_by_item,
        "item_menu_hint": item_menu_hint,
        "item_names_by_id": item_names_by_id,       # NEW
        "aliases_by_item": aliases_by_item,         # NEW
        "category_by_item": category_by_item,       # NEW
        "beverage_item_ids": beverage_item_ids,     # NEW (set)
        "specials_block_combo": specials_block_combo,  # NEW (set)
        "_menus": {  # keep raw menus for name lookup / post-cancel checks
            "american": bundle.get("american") or {},
            "middle-eastern": bundle.get("middle-eastern") or {},
        },
    }


ALLOWED = {"item_id","qty","variant_id","modifiers","attributes","combo_opt_in","menu_hint"}

def _coerce_notes_only(attrs: dict) -> dict:
    if not isinstance(attrs, dict):
        return {}
    notes = attrs.get("_notes")
    if notes is None:
        return {}
    if isinstance(notes, str):
        notes = [notes]
    # trim, dedupe, cap length of each note
    clean, seen = [], set()
    for n in notes:
        s = " ".join(str(n).split())[:80]
        k = s.lower()
        if s and k not in seen:
            clean.append(s)
            seen.add(k)
    return {"_notes": clean} if clean else {}

def _normalize_to_lines(cart_lines: list) -> list[Line]:
    norm: list[Line] = []
    for ln in cart_lines or []:
        if isinstance(ln, Line):
            # still sanitize attributes to notes-only
            ln.attributes = _coerce_notes_only(getattr(ln, "attributes", {}) or {})
            norm.append(ln)
            continue
        try:
            d = {k: v for k, v in dict(ln).items() if k in ALLOWED}
            # collapse duplicate modifiers as before
            mods_accum = {}
            for m in (d.get("modifiers") or []):
                mid = m.get("modifier_id")
                if not mid:
                    continue
                # NEW: ignore combo-like modifiers at normalization time
                if COMBO_MOD_RE.search(str(mid)):
                    continue
                qty = int(m.get("qty", 1))
                mods_accum[mid] = mods_accum.get(mid, 0) + qty

            d["modifiers"] = [Mod(modifier_id=k, qty=v) for k, v in mods_accum.items()]

            d.setdefault("variant_id", "base")
            d.setdefault("combo_opt_in", False)
            d["attributes"] = _coerce_notes_only(d.get("attributes") or {})
            norm.append(Line(**d))
        except ValidationError:
            continue
    return norm


def _json_only(text: str) -> Dict[str, Any]:
    """extract first {...} JSON object from text; tolerate trailing commas"""
    if not text:
        return {}
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {}
    blob = text[s:e+1]
    try:
        return json.loads(blob)
    except Exception:
        blob = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            return json.loads(blob)
        except Exception:
            return {}

# --- status detection helpers ---
CANCEL_WORDS = [
    r"\bcancel\b", r"\bcancel it\b", r"\bcancel the order\b", r"\bnevermind\b",
    r"\bnever mind\b", r"\bforget it\b", r"\bno order\b", r"\bdon't place it\b",
    r"\bi changed my mind\b", r"\bi don't want it\b"
]
# CANCEL_RE = re.compile("|".join(CANCEL_WORDS), re.I)

INCOMPLETE_DELIVERY_RE = re.compile(r"\b(deliver|delivery)\b", re.I)
ADDRESS_HINT_RE = re.compile(r"\b(\d{2,5}\s+[\w\s\.\-]+(ave|avenue|st|street|rd|road|blvd|lane|ln|dr|drive|apt|apartment|suite|#)\b.*)", re.I)

CANCEL_RE = re.compile(r"\b(cancel|never\s*mind|forget\s*it|no\s*order)\b", re.I)

def _detect_cancellation(t: str) -> bool:
    return bool(CANCEL_RE.search(t or ""))

def _needs_address(collected: Dict[str, Any], transcript: str) -> bool:
    order_type = (collected.get("order_type") or "").lower()
    # if it’s a delivery and no address captured, mark as pending
    if order_type == "delivery":
        addr = (collected.get("address") or "").strip()
        return not addr
    return False



YES_WORDS = {"yes","yeah","yup","sure","ok","okay","pls","please","y"}
OFFER_PAT = r"(combo|fries\s+and\s+a?\s*can\s+of?\s*soda|fries\s+and\s+soda)"

def _force_combo_if_affirmed(transcript: str, cart_lines: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    t = (transcript or "").lower()
    OFFER_PAT = r"(make it a combo|combo|fries\s+and\s+a?\s*can\s*(of)?\s*soda)"
    YES_WORDS = {"yes","yeah","yup","sure","ok","okay","pls","please","y"}
    NO_WORDS = {"no","nah","nope"}

    if re.search(OFFER_PAT, t):
        if any(w in t for w in NO_WORDS):
            for ln in cart_lines:
                ln["combo_opt_in"] = False
            return cart_lines
        if any(w in t for w in YES_WORDS):
            mains = [ln for ln in cart_lines if not ln["item_id"].startswith(("bev-","side-","drink-"))]
            if mains:
                mains[-1]["combo_opt_in"] = True
    return cart_lines


def _build_name_index(menu_bundle: dict) -> dict:
    idx = {}
    for hint in ("american", "middle-eastern"):
        menu = menu_bundle.get(hint) or {}
        for cat in (menu.get("categories") or []):
            for it in (cat.get("items") or []):
                iid = it.get("id")
                if iid:
                    idx[(iid, hint)] = it.get("name") or iid
    return idx


def _summarize_menu_for_llm(menu_doc: dict, max_chars: int = 8000) -> str:
    """compact, token-friendly menu reference"""
    lines = []
    for cat in (menu_doc.get("categories") or []):
        cname = cat.get("name") or cat.get("id", "")
        lines.append(f"{cname}:")
        for it in (cat.get("items") or []):
            iid = it.get("id", "")
            price = it.get("price")
            if price is None and isinstance(it.get("variants"), list):
                # pick base variant price if present
                base = next((v for v in it["variants"] if v.get("id")=="base"), None)
                price = base.get("price") if base else None
            price_str = f"${price:.2f}" if isinstance(price, (int,float)) else "-"
            lines.append(f" - {iid} ({price_str})")
    out = "\n".join(lines)
    return out[:max_chars]

RICE_RE = re.compile(r"\b(yellow|brown)\s+rice\b", re.I)
def _resolve_status(
    transcript: str,
    extracted_lines: list[dict],
    transfer_info: dict | None,
    cancel_seen: bool,
    call_cut_seen: bool,
) -> str:
    order_seen = bool(extracted_lines)
    if cancel_seen:
        return "canceled"
    if transfer_info:
        return "transfer_requested"
    if call_cut_seen and not order_seen:
        return "incomplete"
    if order_seen:
        return "confirmed"
    return "no_order"


def _resolve_status_and_pricing_visibility(
    *,
    transcript: str,
    collected: Dict[str, Any],
    schema_ctx: dict,
    extracted_lines: list[dict],
    priced_lines_exist: bool,
    transfer_info: dict | None,
    cancel_seen: bool,
    call_cut_seen: bool,
    info_only: bool,
) -> tuple[str, bool]:
    """
    Returns (final_status, show_pricing_bool)
    Deterministic logic: pricing only if order exists.
    """
    order_seen = bool(extracted_lines)
    final_cancel = _last_intent_wins(transcript, schema_ctx, extracted_lines, cancel_seen)
    delivery_missing = (
        (collected.get("order_type") or "").lower() == "delivery"
        and not (collected.get("address") or "").strip()
    )

    # unified deterministic matrix
    if order_seen and not final_cancel:
        status = "pending_address" if delivery_missing else "confirmed"
    elif transfer_info:
        status = "transfer_requested"
    elif final_cancel:
        status = "canceled"
    elif call_cut_seen:
        status = "incomplete"
    elif info_only:
        status = "info"
    else:
        status = "no_order"

    # pricing: only if actual priced items exist
    show_pricing = bool(priced_lines_exist)

    return status, show_pricing

# ALSHAM_RE = re.compile(r"\b(al[-\s]?sham|alsham)\b", re.I)
# def _inject_alsham_style(transcript: str, lines: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
#     if not ALSHAM_RE.search(transcript or ""):
#         return lines
#     out = []
#     for ln in lines:
#         d = dict(ln)
#         if d.get("item_id","").startswith("ny-"):
#             d["variant_id"] = "al_sham"
#         attrs = dict(d.get("attributes") or {})
#         notes = attrs.get("_notes") or []
#         if isinstance(notes, str):
#             notes = [notes]
#         notes.append("Al-Sham style")
#         attrs["_notes"] = list(set(notes))
#         d["attributes"] = attrs
#         out.append(d)
#     return out

def _repair_alias_map(amap: dict | None) -> dict:
    """Fix bad/stale alias keys without touching DB."""
    fixes = {
        "bg-burger-single": "bg-burger",  # not an item; map to real one
        "bg-spicy-chicken-burger": "hg-spicy-chicken",  # closest real spicy chicken “sandwich”
        "sp-2-chicken-sandwiches-fries": "sp-5-2-chicken-sandwiches-fries",  # real ID
    }
    out: dict[str, list[str]] = {}
    for k, phrases in (amap or {}).items():
        target = fixes.get(k, k)
        out.setdefault(target, []).extend(phrases or [])
    return out

def _build_alias_index_from_bundle(bundle: dict) -> dict[str, dict]:
    """
    Returns phrase(lower) -> {item_id, menu_hint}
    Merges both menus, repairs alias keys.
    """
    idx: dict[str, dict] = {}
    for hint in ("american", "middle-eastern"):
        amap = ((bundle.get(hint) or {}).get("alias_map")) or {}
        amap = _repair_alias_map(amap)
        for iid, phrases in amap.items():
            for p in (phrases or []):
                if not p: continue
                key = p.lower().strip()
                if key and key not in idx:
                    idx[key] = {"item_id": iid, "menu_hint": hint}
    return idx

def _seed_cart_from_aliases(transcript: str, alias_idx: dict[str, dict]) -> list[dict]:
    """Find simple substring matches; 1× base, no mods, no combo. De-duped by item_id."""
    low = (transcript or "").lower()
    seen: set[str] = set()
    seeds: list[dict] = []
    for phrase, meta in alias_idx.items():
        if phrase in low and meta["item_id"] not in seen:
            seeds.append({
                "item_id": meta["item_id"],
                "qty": 1,
                "variant_id": "base",
                "combo_opt_in": False,
                "modifiers": [],
                "attributes": {},
                "menu_hint": meta["menu_hint"],
            })
            seen.add(meta["item_id"])
    return seeds


def _inject_rice_attribute(transcript: str, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    m = RICE_RE.search(transcript or "")
    if not m:
        return lines
    rice = m.group(1).lower()
    out = []
    for ln in lines:
        d = dict(ln)
        attrs = dict(d.get("attributes") or {})
        if d.get("item_id", "").startswith("ny-"):  # only apply to NY chicken type items
            attrs["rice_type"] = rice
        d["attributes"] = attrs
        out.append(d)
    return out

COMBO_ANY_RE   = re.compile(r"\bcombo\b", re.I)
COMBO_ALL_RE   = re.compile(r"\b(both|all)\b.*\bcombo\b|\bcombo\b.*\b(both|all)\b", re.I)
COMBO_ONE_RE   = re.compile(r"\b(one|1)\b.*\bcombo\b|\bcombo\b.*\b(one|1)\b|\bjust\s+one\s+of\s+them\b", re.I)
COMBO_NO_RE    = re.compile(r"\b(no|nah|nope)\b.*\bcombo\b|\bwithout\b.*\b(combo|fries|soda)\b", re.I)

def _is_main_line(ln: dict) -> bool:
    iid = (ln.get("item_id") or "")
    # treat beverages/sides as non-mains generically by id prefixes you already use
    return not iid.startswith(("bev-","drink-","side-"))

def _reconcile_combo(transcript: str, cart_lines: list[dict]) -> list[dict]:
    """
    Universal, context-aware combo assignment:
    - 'both/all combo' => set combo on all mains.
    - 'one combo' or 'just one of them' => set combo on the most recent main only.
    - 'no combo' => force all mains combo=false.
    - If no explicit combo language, leave model’s flags as-is.
    """
    t = (transcript or "").lower()

    # Hard declines win
    if COMBO_NO_RE.search(t):
        for ln in cart_lines:
            if _is_main_line(ln):
                ln["combo_opt_in"] = False
        return cart_lines

    # All/both => set all mains
    if COMBO_ALL_RE.search(t):
        for ln in cart_lines:
            if _is_main_line(ln):
                ln["combo_opt_in"] = True
        return cart_lines

    # Exactly one => set most recent main only
    if COMBO_ONE_RE.search(t) and COMBO_ANY_RE.search(t):
        for ln in reversed(cart_lines):
            if _is_main_line(ln):
                ln["combo_opt_in"] = True
                break
        # Do not unset others if user didn’t say “only one”; leave model’s prior flags intact
        return cart_lines

    # If there is any explicit 'combo' mention without number,
    # leave the model’s per-line decisions as-is (the LLM may already have split lines).
    return cart_lines

# COMBO_MOD_ID_RE = re.compile(r"(combo|fries.*(can|soda)|fries.*drink|make.*combo)", re.I)
#
# def _hoist_combo_mod_to_flag(lines: List[Dict]) -> List[Dict]:
#     out = []
#     for ln in (lines or []):
#         mods, combo_seen = [], bool(ln.get("combo_opt_in", False))
#         for m in (ln.get("modifiers") or []):
#             mid = (m.get("modifier_id") or "")
#             if COMBO_MOD_ID_RE.search(mid):  # modifier -> flag
#                 combo_seen = True
#                 continue
#             mods.append(m)
#         ln["combo_opt_in"] = combo_seen
#         ln["modifiers"] = mods
#         out.append(ln)
#     return out


def _extract_cart(transcript_user: str, transcript_with_agent: str | None, schema_ctx: dict) -> Dict[str, Any]:
    """
    Universal extractor — model decides items only using known menu names/aliases.
    Global combo handling included. No static per-scenario rules.
    """
    item_ids = schema_ctx["allowed_item_ids"]
    modifier_ids = schema_ctx["allowed_modifier_ids"]
    variants_by_item = schema_ctx["variants_by_item"]
    item_menu_hint = schema_ctx["item_menu_hint"]
    item_names_by_id = schema_ctx["item_names_by_id"]
    aliases_by_item = schema_ctx["aliases_by_item"]

    system_prompt = """
You are a RESTAURANT ORDER EXTRACTOR.

You will receive the FULL conversation, including both Agent and User lines.
Use Agent lines ONLY as context (offers, clarifications). Extract items STRICTLY
from what the USER actually requests or explicitly accepts. Never invent items.

!!! OUTPUT FORMAT (JSON ONLY) !!!
Return ONLY a single JSON object with this exact shape (no prose, no markdown):

{
  "cart_lines": [
    {
      "item_id": "<from allowed_item_ids>",
      "qty": <int>=1+,
      "variant_id": "<from variants_by_item[item_id] or 'base'>",
      "combo_opt_in": <true|false>,
      "modifiers": [ { "modifier_id": "<from allowed_modifier_ids>", "qty": <int>=1+ } ],
      "attributes": { "_notes": ["<verbatim user phrases, ≤6 words each>"] },
      "menu_hint": "<american|middle-eastern>"
    }
  ],
  "reason": "<<=160 chars, or empty string>"
}

If nothing valid is ordered, return:
{ "cart_lines": [], "reason": "No valid order" }

CONTEXT YOU WILL RECEIVE (as JSON fields):
- allowed_item_ids
- item_names_by_id
- aliases_by_item
- modifier_ids (allowed_modifier_ids)
- variants_by_item
- item_menu_hint
- user_transcript   (full convo text; prefix lines with 'User:' or 'Agent:')

HARD RULES (STRICT):
1) Match only when a USER utterance contains the item’s exact name or an approved alias.
   - Agent suggestions DO NOT count unless followed by an explicit USER acceptance (e.g., “yes, add that”).
2) Do NOT infer items from ingredients, category names, or general intent.
3) Do NOT output any item_id, variant_id, or modifier_id that is not present in the allowed lists.
4) qty must be an integer ≥1. If user says a bare number before/after the item, use that as qty.
5) attributes must ONLY contain `_notes` (array of short verbatim user phrases, each ≤6 words). No other keys.
6) menu_hint MUST be set from item_menu_hint[item_id] exactly.
7) Never add combo as a modifier. combo_opt_in is the ONLY combo signal in this JSON.
8) Never output prices, currency, totals, or any fields outside the schema.
9) • You will be given both transcripts:
   - transcript_user: ONLY user lines (authoritative for items/combos)
   - transcript_with_agent: full convo (Agent lines allowed ONLY to resolve pronouns like “that one”, never to add items or combos by themselves).
10) Strictly matched the extracted item with the menu.
COMBO (STRICT):
- Set combo_opt_in=true ONLY if the USER explicitly requests/accepts combo in their own words:
  examples: “make it a combo”, “with fries and soda”, “yes, combo”, “okay, add fries and a soda”.
- If the Agent offers a combo, you must see a USER acceptance (“yes/ok/sure/please” OR a rephrasing that clearly accepts).
- If the USER declines (“no combo”, “just regular”, “without fries/soda”), set combo_opt_in=false.
- If the USER mentions combo for one of multiple mains, apply true only to that referenced main. If ambiguous, apply to the LAST mentioned main. Do NOT set combo on all by default.
- Agent offers do NOT count unless followed by explicit USER acceptance in transcript_user.


CUSTOMER PHRASE & WORD ORDER AWARENESS (CRITICAL)

Customers often say item names in different word orders or with small phrasing changes (e.g., “burger chicken deluxe” instead of “Deluxe Chicken Burger”).

You must match meaning and tokens, not position — recognize valid items even when the customer rearranges words or adds small fillers.

However, only match if all key tokens of an item’s official name or alias appear together in the USER’s phrase (order may vary, but words must all be present).

Do not guess based on partial overlap or ingredients alone.

Always double-check the full phrase before extracting — a single wrong item  can ruin the entire order.

When multiple candidates share words, choose the item whose name best matches the USER’s exact spoken wording, including modifiers like “deluxe”, “large”, “combo”, or “spicy”.
APPETIZER QUANTITY & PACKS:
- If the USER gives a count without “pack/pcs/pieces/deal/combo”, treat as singles: qty=<count>, variant_id="single" if available else "base".
- Use an explicit pack variant ONLY if the USER clearly says “pack/deal/4-piece/5 pcs/…”.
- Never add a generic placeholder (e.g., “Appetizer Single”) when a concrete item (e.g., “Chicken Samosa”) was recognized; prefer the concrete item and drop placeholders.

MULTIPLE LINES:
- Split lines when the USER differentiates items (e.g., “one combo, one regular”).
- Different variants, attributes, or combo flags → separate lines.

NEGATIVE RULES:
- Do NOT rely on Agent lines alone to add anything.
- Do NOT carry over items suggested but never accepted by the USER.
- Do NOT guess drink flavors or sides unless the USER said them; if unspecified, omit from notes.

VALIDATION BEFORE RETURN:
- Each cart line must pass:
  • item_id ∈ allowed_item_ids
  • variant_id ∈ variants_by_item[item_id] (else use "base")
  • Every modifier.modifier_id ∈ allowed_modifier_ids (drop invalid ones)
  • qty ≥ 1 (integer)
  • menu_hint = item_menu_hint[item_id]
- If no valid lines remain, return { "cart_lines": [], "reason": "No valid order" }.

MICRO EXAMPLES (NOT PART OF OUTPUT):
- User: “NY over rice. Make it a combo.” → combo_opt_in=true for that main.
- Agent: “Want to make it a combo?” User: “No, regular.” → combo_opt_in=false.
- User: “5 samosas” (no “pcs/pack”) → qty=5, variant_id="single" (if exists) else "base".
- User: “Two chicken sandwiches; make ONLY one a combo.” → last mentioned main combo_opt_in=true; the other false.

Return ONLY the JSON object described above. No explanations.


"""

    user_prompt = json.dumps({
        "transcript_user": transcript_user,
       "transcript_with_agent": (transcript_with_agent or ""),
        "allowed_item_ids": item_ids,
        "item_names_by_id": item_names_by_id,
        "aliases_by_item": aliases_by_item,
        "modifier_ids": modifier_ids,
        "variants_by_item": variants_by_item,
        "item_menu_hint": item_menu_hint,
    }, indent=2)

    try:
        rsp = client.chat.completions.create(
            model=EXTRACT_MODEL,
            temperature=0,
            top_p=1,
            max_tokens=1000,
            response_format={"type": "json_object"},  # NEW
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = rsp.choices[0].message.content.strip()
        data = _json_only(raw) or {}
    except Exception as e:
        log.error(f"Extractor LLM error: {e}")
        return {"cart_lines": [], "reason": f"LLM extraction failed: {e}"}

    # --- sanitize model output ---
    sanitized = []
    for ln in (data.get("cart_lines") or []):
        if not isinstance(ln, dict):
            continue

        iid = str(ln.get("item_id") or "").strip()
        if iid not in item_ids:
            continue

        # quantity
        try:
            qty = max(1, int(ln.get("qty", 1) or 1))
        except Exception:
            qty = 1

        # variant check
        vmap = variants_by_item.get(iid) or ["base"]
        vid = ln.get("variant_id") or "base"
        if vid not in vmap:
            vid = "base"


        # modifiers
        clean_mods = []
        for m in (ln.get("modifiers") or []):
            if not isinstance(m, dict):
                continue
            mid = str(m.get("modifier_id") or "").strip()
            # NEW: drop combo-like modifiers early
            if COMBO_MOD_RE.search(mid):
                continue
            if mid in modifier_ids:
                try:
                    mqty = max(1, int(m.get("qty", 1)))
                except Exception:
                    mqty = 1
                clean_mods.append({"modifier_id": mid, "qty": mqty})

        # attributes
        attrs = ln.get("attributes") or {}
        if isinstance(attrs, str):
            attrs = {"_notes": [attrs]}
        elif not isinstance(attrs, dict):
            attrs = {}
        if "_notes" in attrs and not isinstance(attrs["_notes"], list):
            attrs["_notes"] = [str(attrs["_notes"])]

        sanitized.append({
            "item_id": iid,
            "qty": qty,
            "variant_id": vid,
            "combo_opt_in": bool(ln.get("combo_opt_in", False)),
            "modifiers": clean_mods,
            "attributes": attrs,
            "menu_hint": item_menu_hint.get(iid, "middle-eastern"),
        })

    # 1) Category policies (generic, menu-driven)
    sanitized = _apply_category_policies(transcript_user, sanitized, schema_ctx)

    # 2) Prefer specific over placeholders (menu-driven)
    placeholders = _index_placeholders(schema_ctx.get("_menus") or {})
    sanitized = _prefer_specific_over_placeholder(sanitized, schema_ctx, placeholders)

    # 3) Dedupe logically identical lines (you already have this)
    sanitized = _merge_duplicate_lines(sanitized)

    # 4) Ensure priceable (deterministic repair or drop)
    sanitized, _dbg_unpriceable = _priceability_gate(sanitized, schema_ctx)


    sanitized = _reconcile_combo(transcript_user, sanitized)

    # sanitized = _hoist_combo_mod_to_flag(sanitized)

    # sanitized = _normalize_appetizer_packs(transcript, sanitized, schema_ctx)
    # sanitized = _drop_generic_appetizer_placeholders(sanitized)
    # sanitized = _merge_duplicate_lines(sanitized)

    # --- enforce canonical menu hint ---
    for ln in sanitized:
        iid = ln["item_id"]
        hint = item_menu_hint.get(iid, ln.get("menu_hint") or "middle-eastern")
        ln["menu_hint"] = "american" if hint == "american" else "middle-eastern"

    return {"cart_lines": sanitized, "reason": data.get("reason", "")}


def _summarize(
    pricing: Dict[str, Any],
    collected: Dict[str, Any],
    *,
    call_meta: Dict[str, Any] | None = None,
    transcript: str | None = None,
) -> str:
    """
    Unified summary generator with optional LLM polish.
    - Deterministic base summary from pricing + metadata.
    - Then uses an LLM post-pass to improve readability (no changes to numbers or order details).
    """
    import json, re
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    transcript = transcript or ""
    call_meta = (call_meta or {}).copy()

    def _json_safe(o):
        try:
            json.dumps(o); return o
        except Exception:
            return str(o)

    # ---------- STATUS LINE ----------
    def _status_line(cm: Dict[str, Any]) -> str:
        st = (cm.get("status") or "confirmed").lower()
        if st == "canceled":
            return "Status: Canceled by caller"
        if st == "pending_address":
            return "Status: Incomplete — address needed"
        if st in {"call_cut", "incomplete"}:
            return "Status: Call ended before confirming any order"
        if st == "transfer_requested":
            tgt = (cm.get("transfer") or {}).get("target") or "human_agent"
            return f"Status: Call transferred to {tgt.replace('_',' ')} (requested by caller)"
        if st == "info":
            return "Status: Info-only call — no order placed"
        if st == "no_order":
            return "Status: No order placed"
        return "Status: Confirmed"

    # ---------- HEADER ----------
    ot = (collected.get("order_type") or call_meta.get("order_type") or "unspecified").title()
    name = (collected.get("customer_name") or call_meta.get("customer_name") or "").strip()
    phone = (collected.get("phone") or call_meta.get("phone") or "").strip()
    address = (collected.get("address") or call_meta.get("address") or "").strip()

    parts: list[str] = []
    parts.append(_status_line(call_meta))
    parts.append(f"Order: {ot}")
    if name or phone:
        parts.append(f"Customer: {name or '—'}{' • ' + phone if phone else ''}")
    if ot.lower() == "delivery":
        parts.append(f"Address: {address or '—'}")

    # ---------- NO ITEMS ----------
    # Only show lines that actually have a priced total (> 0)
    lines = [ln for ln in (pricing.get("lines") or []) if float(ln.get("line_total") or 0) > 0]
    if not lines:
        st = call_meta.get("status", "").lower()
        if st == "canceled":
            parts.append("Order canceled by caller.")
        elif st == "pending_address":
            parts.append("Incomplete order — address needed.")
        elif st == "info":
            parts.append("Info-only call — no order placed.")
        elif st == "transfer_requested":
            tgt = (call_meta.get("transfer") or {}).get("target") or "human_agent"
            parts.append(f"Call transferred to {tgt.replace('_',' ')} (requested by caller).")
        else:
            parts.append("No order placed.")
        summary = "\n".join(parts)
    else:
        # ---------- ITEMS ----------
        for ln in lines:
            qty = int(ln.get("qty", 1))
            desc = ln.get("desc") or ln.get("item_id", "?")
            variant = (ln.get("variant_id") or "base")
            item_str = f"- {qty}× {desc}"
            if variant and variant != "base":
                item_str += f" [{variant}]"

            combo_total = float(ln.get("combo_total", 0.0) or 0.0)
            line_total = float(ln.get("line_total", 0.0) or 0.0)
            if line_total > 0:
                item_str += f" — ${line_total:.2f}"
            else:
                item_str += " (price missing)"

            attrs = ln.get("attributes") or {}
            bits = []
            for raw in (attrs.get("_notes") or []):
                s = " ".join(str(raw).split())
                if s:
                    bits.append(s)
            if bits:
                item_str += f" ({', '.join(bits)})"

            if combo_total > 0:
                item_str += " — Combo added"
            parts.append(item_str)

        # ---------- TOTALS ----------
        subtotal = float(pricing.get("subtotal") or 0.0)
        tax = float(pricing.get("tax") or 0.0)
        total = float(pricing.get("total") or (subtotal + tax))
        include_tax = bool(pricing.get("include_tax", True))
        parts.append(f"Subtotal: ${subtotal:.2f}")
        if include_tax:
            parts.append(f"Tax: ${tax:.2f}")
        parts.append(f"Total: ${total:.2f}")

        # ---------- NOTES ----------
        note_bits = []
        for ln in lines:
            raw = (ln.get("attributes") or {}).get("_notes") or []
            if isinstance(raw, str):
                raw = [raw]
            for r in raw:
                s = " ".join(str(r).split())
                if s and s.lower() not in {n.lower() for n in note_bits}:
                    note_bits.append(s)
        if note_bits:
            parts.append("Notes: " + ", ".join(note_bits))

        summary = "\n".join(parts)

    # ---------- LLM POLISH (safely rephrases) ----------
    try:
        sys_prompt = (
            "You are a careful editor for restaurant order summaries.\n"
            "Rephrase the following summary for clarity and flow, but:\n"
            "- DO NOT change or remove any item names, quantities, prices, or totals.\n"
            "- Keep all dollar amounts, numbers, and combo lines identical.\n"
            "- Preserve the overall structure (Status, Order, Customer, Items, Totals, Notes).\n"
            "If everything is already fine, return it unchanged."
        )
        user_prompt = f"Here is the summary:\n\n{summary}"

        rsp = client.chat.completions.create(
            model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini"),
            temperature=0,
            top_p=1,
            seed=17,
            max_tokens=420,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        polished = (rsp.choices[0].message.content or "").strip()

        # Safety: ensure prices and qtys match
        old_nums = set(re.findall(r"\$\d+\.\d{2}|\d+×", summary))
        new_nums = set(re.findall(r"\$\d+\.\d{2}|\d+×", polished))
        if old_nums == new_nums:
            return polished
    except Exception:
        pass

    return summary


# @router.post("/ingest_eleven")
@router.post("/ingest_eleven")
def ingest_eleven(payload: Dict[str, Any], request: Request):
    # transcript_text = _flatten_eleven_transcript(payload, include_agent=True)
    t_user_only = _flatten_eleven_transcript(payload, include_agent=False)
    t_with_agent = _flatten_eleven_transcript(payload, include_agent=True)
    customer_name = _extract_name(payload)
    phone = _get_caller_phone(payload)
    order_type = _detect_order_type(t_user_only)
    collected = {
        "customer_name": customer_name or "",
        "phone": phone or "",
        "order_type": order_type,
        "menu_hint": "middle-eastern",
    }

    bundle = _load_menu_bundle(collected.get("menu_hint"))
    schema_ctx = _union_schema_from_bundle(bundle)

    extraction = _extract_cart(t_user_only, t_with_agent, schema_ctx)
    raw_lines: List[Dict[str, Any]] = _repair_llm_lines(extraction.get("cart_lines") or [])
    line_models = _normalize_to_lines(raw_lines)

    cancel_seen = _detect_cancellation(t_user_only)
    transfer_info_detected = _detect_transfer_both_sides(t_with_agent, t_user_only)
    transfer_info = transfer_info_detected or _handoff_from_payload(payload)
    # transfer_info = _detect_transfer(t_user_only) or _handoff_from_payload(payload)
    call_cut_seen = _detect_call_cut(t_user_only)
    info_only = _detect_info_only(t_user_only) and not line_models

    pricing = None
    priced_lines_exist = False
    if line_models:
        try:
            pricing = calc_core(
                payload_cart=line_models,
                payload_utterance=t_user_only,
                include_tax=True,
                menu_bundle=bundle,
                combo_mode="explicit",
                allow_auto_combo=False
            )
        except TypeError:
            pricing = calc_core(
                payload_cart=line_models,
                payload_utterance=t_user_only,
                include_tax=True,
                menu_bundle=bundle,
            )
        # NEW: enforce in all cases
        pricing = _enforce_explicit_combos(pricing, line_models)


        priced_lines_exist = bool((pricing.get("lines") or []))

        # ID→Name
        name_idx = _build_name_index(bundle)
        for ln in (pricing.get("lines") or []):
            iid = (ln.get("src") or {}).get("item_id") or ln.get("item_id")
            hint = ln.get("menu_hint") or (ln.get("src") or {}).get("menu_hint") or "middle-eastern"
            if iid:
                ln["desc"] = name_idx.get((iid, hint), ln.get("desc", iid))

    final_status, show_pricing = _resolve_status_and_pricing_visibility(
        transcript=t_user_only,
        collected=collected,
        schema_ctx=schema_ctx,
        extracted_lines=raw_lines,
        priced_lines_exist=bool(pricing and (pricing.get("lines") or [])),
        transfer_info=transfer_info,
        cancel_seen=cancel_seen,
        call_cut_seen=call_cut_seen,
        info_only=info_only,
    )
    # final_status = _resolve_status(
    #     transcript=t_user_only,
    #     extracted_lines=raw_lines,
    #     transfer_info=transfer_info,
    #     cancel_seen=cancel_seen,
    #     call_cut_seen=call_cut_seen,
    # )

    call_meta = {
        "order_type": (collected.get("order_type") or "unspecified"),
        "customer_name": (collected.get("customer_name") or "").strip(),
        "phone": (collected.get("phone") or "").strip(),
        "address": (collected.get("address") or "").strip(),
        "status": final_status,
    }
    if transfer_info:
        call_meta["transfer"] = transfer_info

    # always nullify if no lines (deterministic)
    # pricing_to_return = pricing if (pricing and (pricing.get("lines") or [])) else None
    pricing_to_return = pricing if (pricing and show_pricing and (pricing.get("lines") or [])) else None
    has_order = bool(pricing_to_return)

    # pricing_to_return = pricing if (pricing and show_pricing) else None
    # has_order = bool(pricing_to_return and (pricing_to_return.get("lines") or []))

    summary = _summarize(pricing_to_return or {}, collected, call_meta=call_meta, transcript=t_with_agent)

    if "text/plain" in (request.headers.get("accept","").lower()):
        return PlainTextResponse(summary)

    return {
        "ok": True,
        "has_order": has_order,
        "pricing": pricing_to_return,
        "summary": summary,
        "extraction": extraction,
        "collected": collected,
        "final_status": final_status,
    }
