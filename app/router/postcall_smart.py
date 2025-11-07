# app/router/rest_ingest_v2.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
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



NAME_RE = re.compile(r"The user,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
PICKUP_RE = re.compile(r"\b(pick\s*up|pickup|pick-up)\b", re.I)
DELIVERY_RE = re.compile(r"\bdeliver(y)?\b", re.I)

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


import logging
log = logging.getLogger(__name__)

def _union_schema_from_bundle(bundle: dict) -> dict:
    item_ids = []
    modifier_ids = []
    variants_by_item = {}
    item_menu_hint = {}

    seen = {}  # iid -> hint_key (first seen)

    for hint_key in ("american", "middle-eastern"):
        menu_doc = bundle.get(hint_key) or {}
        # global modifiers
        for m in (menu_doc.get("modifier_catalog") or []):
            mid = m.get("id")
            if mid:
                modifier_ids.append(mid)

        for cat in (menu_doc.get("categories") or []):
            # items
            for it in (cat.get("items") or []):
                iid = it.get("id")
                if not iid:
                    continue

                # 🔔 collision detection
                if iid in seen and seen[iid] != hint_key:
                    log.warning(
                        "Duplicate item_id across menus detected: %s  (first=%s, now=%s)",
                        iid, seen[iid], hint_key
                    )
                    # optional: raise AssertionError to fail-fast in dev
                    # assert False, f"Duplicate item_id across menus: {iid}"

                seen.setdefault(iid, hint_key)

                item_ids.append(iid)
                item_menu_hint[iid] = hint_key

                if isinstance(it.get("variants"), list) and it["variants"]:
                    variants_by_item[iid] = [v.get("id") for v in it["variants"] if v.get("id")]
                else:
                    variants_by_item[iid] = ["base"]

            # category modifiers
            for mod in (cat.get("category_modifiers") or []):
                if isinstance(mod, dict) and mod.get("id"):
                    modifier_ids.append(mod["id"])
                elif isinstance(mod, str):
                    modifier_ids.append(mod)

            for ex in (cat.get("extras") or []):
                mid = ex.get("id")
                if mid:
                    modifier_ids.append(mid)

    item_ids = sorted(set(item_ids))
    modifier_ids = sorted(set(modifier_ids))

    return {
        "allowed_item_ids": item_ids,
        "allowed_modifier_ids": modifier_ids,
        "variants_by_item": variants_by_item,
        "item_menu_hint": item_menu_hint,
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


# def _normalize_to_lines(cart_lines: list) -> list[Line]:
#     """Convert list of dicts from LLM into strong Line models for calc_core."""
#     norm: list[Line] = []
#     ALLOWED = {"item_id","qty","variant_id","modifiers","attributes","combo_opt_in","menu_hint"}
#     for ln in cart_lines or []:
#         if isinstance(ln, Line):
#             norm.append(ln)
#             continue
#         try:
#             d = {k: v for k, v in dict(ln).items() if k in ALLOWED}  # ← extras strip
#             mods_accum = {}
#             for m in (d.get("modifiers") or []):
#                 mid = m.get("modifier_id")
#                 if not mid:
#                     continue
#                 qty = int(m.get("qty", 1))
#                 mods_accum[mid] = mods_accum.get(mid, 0) + qty
#             d["modifiers"] = [Mod(modifier_id=k, qty=v) for k, v in mods_accum.items()]
#
#             # sensible defaults
#             d.setdefault("variant_id", "base")
#             d.setdefault("combo_opt_in", False)
#             d.setdefault("attributes", {})
#             norm.append(Line(**d))
#         except ValidationError:
#             continue
#     return norm
# ---------- tiny utils ----------
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
    if not (re.search(OFFER_PAT, t) and any(w in t for w in YES_WORDS)):
        return cart_lines
    if any(ln.get("combo_opt_in") for ln in cart_lines):
        return cart_lines
    mains = [ln for ln in cart_lines
             if not str(ln.get("item_id","")).startswith(("side-","bev-","beverage-","drink-"))]
    target = mains[0] if mains else (cart_lines[0] if cart_lines else None)
    if target:
        target["combo_opt_in"] = True
    return cart_lines


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
# ---------- LLM phase 1: extract cart (smart + general) ----------
def _extract_cart(transcript: str, schema_ctx: dict) -> Dict[str, Any]:
    """
    Extract cart using a union of both menus passed in schema_ctx:
    { allowed_item_ids, allowed_modifier_ids, variants_by_item, item_menu_hint }
    """
    item_ids = schema_ctx["allowed_item_ids"]
    modifier_ids = schema_ctx["allowed_modifier_ids"]
    variants_by_item = schema_ctx["variants_by_item"]
    item_menu_hint = schema_ctx["item_menu_hint"]

    system_prompt = '''You are a STRICT restaurant order extractor.

OUTPUT CONTRACT
- Return ONLY valid JSON with keys:
  { "cart_lines": [ {item_id, qty, variant_id, combo_opt_in, modifiers, attributes, menu_hint}... ],
    "reason": "<string>" }
- No prose. No markdown.

ALLOWED IDS
- item_id MUST be one of allowed_item_ids.
- modifiers[*].modifier_id MUST be one of allowed_modifier_ids.
- variant_id MUST be one of variants_by_item[item_id] (use "base" if caller didn’t specify AND "base" exists; else the only variant).

QUANTITY & PACK RULES
- If caller says a number like “7 kibbes” without “pack/4-piece/5-piece/pcs”, set variant_id="single" (or "base" if single not present) and qty=7.
- Only use pack variants if explicitly requested.
- If caller says “singles only / no deals / no bundles”, still extract; the engine will decide bundles.

COMBO ACCEPTANCE (STRICT)
- If a combo is offered and the caller clearly affirms within 2 turns, set combo_opt_in=true for the LAST MAIN item. If declined, keep false.
- When combo_opt_in=true, DO NOT add separate fries/soda lines until customer says i want a fries and i want a can of soda.


ATTRIBUTES (NOTES-ONLY)
- Put the caller’s exact phrases as a list in attributes._notes, e.g. ["no peppers","remove olives","sauce on side"].
- Use verbatim snippets from the caller (≤6 words each). Do NOT paraphrase or generalize.
- Do NOT add or guess any boolean keys (no no_onions, no_tomatoes, etc.).
- If nothing notable, set attributes to {}.

SAUCE GUARD
- NY platters include white & red; add paid modifier only if caller says “extra”.

DISAMBIGUATION
- If ambiguous, omit and put a short reason in "reason".
DRINKS (STRICT)
- Beverage item_ids live only in the American menu (they start with "bev-").
- Extract a drink ONLY if the caller explicitly orders it or clearly accepts with a drink mention.
  Examples of order verbs: "I want Coke", "add a water", "get me a Snapple", "I'll have a can of soda".
- DO NOT extract drinks when the caller is only asking (e.g., "do you have Sprite?", "what drinks do you have?") or when the agent is listing options.
- If a main item is made a combo, do NOT add a separate fries or soda line unless the caller explicitly orders them as separate items.
- If the caller specifies a drink flavor for a combo (e.g., "combo with Sprite"), do not add a separate beverage line; instead, put a short note like "Sprite for combo" in attributes._notes of the combo’d main line.

MENU HINT
- Set menu_hint using item_menu_hint[item_id].

ON FAILURE
- If nothing valid: return { "cart_lines": [], "reason": "No order found" } exactly.'''

    user_prompt = (
        f"Transcript:\n{transcript}\n\n"
        f"allowed_item_ids = {json.dumps(item_ids)}\n"
        f"allowed_modifier_ids = {json.dumps(modifier_ids)}\n"
        f"variants_by_item = {json.dumps(variants_by_item)}\n"
        f"item_menu_hint = {json.dumps(item_menu_hint)}\n\n"
        "Output schema:\n"
        "{ \"cart_lines\": [ { \"item_id\": \"<one of allowed_item_ids>\", \"qty\": 1, "
        "\"variant_id\": \"<from variants_by_item[item_id]>\", \"combo_opt_in\": false, "
        "\"modifiers\": [ {\"modifier_id\": \"<one of allowed_modifier_ids>\", \"qty\": 1} ], "
        "\"attributes\": {}, \"menu_hint\": \"\" } ], \"reason\": \"\" }\n"
        "Return only JSON."
    )

    try:
        rsp = client.chat.completions.create(
            model=EXTRACT_MODEL,
            temperature=0,
            max_tokens=800,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
        )
        raw = (rsp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"cart_lines": [], "reason": f"LLM error: {e}"}

    if raw.lower() in ("null", "none", ""):
        return {"cart_lines": [], "reason": "Model returned null/empty"}

    try:
        data = json.loads(raw)
    except Exception:
        data = _json_only(raw) or {"cart_lines": [], "reason": "Could not parse JSON"}

    if not isinstance(data, dict):
        return {"cart_lines": [], "reason": "Invalid top-level JSON (not an object)"}

    cart = data.get("cart_lines", []) or []
    allowed_items = set(item_ids)
    allowed_mods = set(modifier_ids)

    drop_notes: list[str] = []
    sanitized: list[dict] = []

    for ln in (cart or []):
        if not isinstance(ln, dict):
            drop_notes.append("Non-dict line dropped")
            continue

        # --- item_id
        iid = str(ln.get("item_id", "")).strip()
        if not iid:
            drop_notes.append("Missing item_id")
            continue
        if iid not in allowed_items:
            drop_notes.append(f"Unknown item_id: {iid}")
            continue

        # --- qty (coerce & clamp)
        try:
            qty = int(ln.get("qty", 1))
        except Exception:
            qty = 0
        if qty < 1:
            drop_notes.append(f"Non-positive qty for {iid}")
            continue

        # --- variant
        vmap = variants_by_item.get(iid) or ["base"]
        vid = (ln.get("variant_id") or "base")
        if vid not in vmap:
            vid = "base" if "base" in vmap else vmap[0]

        # --- modifiers (filter to allowed; coerce qty)
        clean_mods = []
        for m in (ln.get("modifiers") or []):
            if not isinstance(m, dict):
                continue
            mid = str(m.get("modifier_id", "")).strip()
            if not mid or mid not in allowed_mods:
                continue
            try:
                mqty = int(m.get("qty", 1))
            except Exception:
                mqty = 1
            if mqty < 1:
                mqty = 1
            clean_mods.append({"modifier_id": mid, "qty": mqty})

        # --- attributes (normalize)
        attrs = ln.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        # force _notes to be a list if present
        if "_notes" in attrs and not isinstance(attrs["_notes"], list):
            attrs["_notes"] = [str(attrs["_notes"])]

        sanitized.append({
            "item_id": iid,
            "qty": qty,
            "variant_id": vid,
            "combo_opt_in": bool(ln.get("combo_opt_in", False)),
            "modifiers": clean_mods,
            "attributes": attrs,
            # menu_hint set below from authoritative map
        })

    # --- force canonical attributes from transcript helpers
    sanitized = _inject_rice_attribute(transcript, sanitized)
    sanitized = _force_combo_if_affirmed(transcript, sanitized)

    # --- HARD ENFORCEMENT: authoritative menu_hint
    for ln in sanitized:
        iid = ln["item_id"]
        hint = item_menu_hint.get(iid, ln.get("menu_hint") or "middle-eastern")
        hint = "american" if hint == "american" else "middle-eastern"
        ln["menu_hint"] = hint

    # --- de-dupe identical logical lines (sum qty)
    def _attr_key(a: dict) -> tuple:
        # stable, hashable key for attributes (order-insensitive)
        items = []
        for k, v in (a or {}).items():
            if k == "_notes":
                items.append((k, tuple(str(x) for x in (v or []))))
            else:
                items.append((k, v if isinstance(v, (str, int, float, bool, type(None))) else str(v)))
        return tuple(sorted(items))

    acc: dict[tuple, dict] = {}
    for ln in sanitized:
        key = (
            ln["item_id"],
            ln["variant_id"],
            ln["menu_hint"],
            _attr_key(ln.get("attributes") or {}),
            tuple(sorted((m["modifier_id"], m["qty"]) for m in ln.get("modifiers") or [])),
            bool(ln.get("combo_opt_in", False)),
        )
        if key not in acc:
            acc[key] = dict(ln)
        else:
            acc[key]["qty"] += ln["qty"]

    cart = list(acc.values())

    # --- final helpers again (in case qty merged etc.)
    cart = _inject_rice_attribute(transcript, cart)
    cart = _force_combo_if_affirmed(transcript, cart)

    reason_bits = [data.get("reason", "")] if data.get("reason") else []
    if drop_notes:
        reason_bits.append(" | ".join(drop_notes))
    reason = " ; ".join([r for r in reason_bits if r])

    return {"cart_lines": cart, "reason": reason or "", "detected_menu": None}


def _summarize(
    pricing: Dict[str, Any],
    collected: Dict[str, Any],
    *,
    call_meta: Dict[str, Any] | None = None,
    transcript: str | None = None,
) -> str:
    """
    Strong, merged summary generator.
    - Handles both: priced orders and no-order/info/canceled calls.
    - Never recomputes math; reads amounts from pricing.
    - Safe JSON serialization for LLM; deterministic fallback.
    """
    import json, re

    def _json_safe(o):
        try:
            json.dumps(o); return o
        except Exception:
            for caster in (dict, list, str):
                try: return caster(o)
                except Exception: pass
            return str(o)

    HINT_RE = re.compile(r"\b(singles?\s+only|no\s+deals|no\s+bundles)\b", re.I)

    def _notes_from_attributes(lines: List[Dict[str, Any]]) -> list[str]:
        notes: list[str] = []
        for ln in lines or []:
            attrs = ln.get("attributes") or {}
            if attrs.get("no_onions") is True: notes.append("no onions")
            if attrs.get("no_tomatoes") is True: notes.append("no tomatoes")
            if attrs.get("extra_white_sauce") is True: notes.append("extra white sauce")
            if attrs.get("extra_red_sauce") is True: notes.append("extra red sauce")
            spice = (attrs.get("spice_level") or "").strip().lower()
            if spice in {"mild","medium","spicy"}: notes.append(spice)
            rice = (attrs.get("rice_type") or "").strip().lower()
            if rice in {"brown","yellow"}: notes.append(f"{rice} rice")
            style = (attrs.get("serving_style") or "").strip().lower()
            if style == "sauce_on_top": notes.append("sauce on top")
            elif style == "meat_on_side": notes.append("meat on the side")
            raw_notes = attrs.get("_notes")
            if isinstance(raw_notes, list):
                notes.extend([str(n) for n in raw_notes if n])
            elif isinstance(raw_notes, str) and raw_notes.strip():
                notes.append(raw_notes.strip())
        # de-dupe keep order
        seen=set(); out=[]
        for n in notes:
            k=n.lower().strip()
            if k not in seen: out.append(n); seen.add(k)
        return out

    def _notes_from_transcript(t: str | None) -> list[str]:
        if not t: return []
        return list({m.lower() for m in HINT_RE.findall(t)})

    def _build_status(cm: Dict[str, Any]) -> str:
        st = (cm.get("status") or "confirmed").lower()
        if st == "canceled": return "Status: Canceled by caller"
        if st in {"pending_address","incomplete"}: return "Status: Incomplete — address needed"
        return "Status: Confirmed"

    transcript = transcript or ""
    call_meta = (call_meta or {}).copy()

    # infer order_type if missing
    if not call_meta.get("order_type"):
        ot = (collected.get("order_type") or "unspecified").lower()
        call_meta["order_type"] = ot if ot in {"pickup","delivery","unspecified"} else "unspecified"
    # identity fallbacks
    call_meta.setdefault("customer_name", (collected.get("customer_name") or "").strip())
    call_meta.setdefault("phone", (collected.get("phone") or "").strip())
    call_meta.setdefault("address", (collected.get("address") or "").strip())
    call_meta.setdefault("status", call_meta.get("status","confirmed"))

    # ---------- NO-ORDER / INFO-ONLY / CANCELED path ----------
    if not (pricing.get("lines") or []):
        sys_prompt = """You are a polite restaurant assistant.
Summarize the call in ONE short sentence (no JSON).
Use exactly one of:
- "Order canceled by caller."
- "Info-only call — no order placed."
- "Call ended before confirming any order."
- "Incomplete order — delivery address missing."
- "No order placed."
Decide from the transcript & context only."""
        user_prompt = (
            f"Transcript tail:\n{(transcript or '')[-1000:]}\n\n"
            f"Collected: {json.dumps(collected, default=_json_safe)}\n"
            f"Meta: {json.dumps(call_meta, default=_json_safe)}"
        )
        try:
            rsp = client.chat.completions.create(
                model=SUMMARY_MODEL,
                temperature=0,
                max_tokens=60,
                messages=[{"role":"system","content":sys_prompt},
                          {"role":"user","content":user_prompt}],
            )
            text = (rsp.choices[0].message.content or "").strip()
            if text: return text
        except Exception:
            pass  # fall back below

        # heuristic fallback
        t = (transcript or "").lower()
        if any(k in t for k in ("cancel", "never mind", "nevermind", "forget it", "no order")):
            return "Order canceled by caller."
        if call_meta.get("order_type") == "delivery" and not call_meta.get("address"):
            return "Incomplete order — delivery address missing."
        if any(k in t for k in ("hour","time","open","close","address","location","where are you",
                                "menu","price","how much","deliver to","delivery radius")):
            return "Info-only call — no order placed."
        return "No order placed."

    # ---------- PRICED ORDER path ----------
    sys = '''You are a precise restaurant agent. Produce a SHORT, human-friendly order recap from given Pricing JSON + metadata.
    — NO markdown, NO bold symbols, NO asterisks.
Use simple plain text formatting only.

HEADER (first):
- Status: Confirmed (unless meta says canceled/incomplete)
- Order: <Pickup|Delivery|Unspecified>
- Name 
- Phone (if present)
- Address (Delivery only)

ITEMS:
- One bullet per line in pricing.lines; "QTY× ITEM [variant if not base] — $LINE_TOTAL"
- If attributes exist: "(yellow rice, spicy, no onions)"
- If modifiers_detail exist: " + <MOD_NAME> ×QTY ($AMT)"
- If combo_total > 0: append " — Combo added"

TOTALS:
- "Subtotal: $X" (+ "Tax: $Y" if include_tax) and "Total: $Z"

NOTES:
NOTES:
- Show one "Notes:" line only if attributes._notes exist in pricing lines.
- Use attributes._notes verbatim; do not infer or add transcript hints.
# - One "Notes:" line if any (attributes booleans/notes + transcript hints like "singles only", "no deals").

STRICTNESS:
- Never recompute math; only use provided amounts.
- Only mention "Combo added" when combo_total > 0.
- No raw JSON; keep within ~10 lines + Notes.'''

    try:
        pricing_json = json.dumps(pricing, default=_json_safe)
        prompt_user = (
            "Render a concise receipt-like summary.\n\n"
            f"Collected: {json.dumps(collected, default=_json_safe)}\n"
            f"Meta: {json.dumps(call_meta, default=_json_safe)}\n"
            f"Pricing JSON:\n{pricing_json}\n\n"
            f"Transcript tail (hints only):\n{(transcript or '')[-600:]}"
        )
        rsp = client.chat.completions.create(
            model=SUMMARY_MODEL,
            temperature=0.2,
            max_tokens=420,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":prompt_user}],
        )
        text = (rsp.choices[0].message.content or "").strip()
        # Remove Markdown bold (**text**) and underscores
        text = re.sub(r'[*_]{2,}', '', text)

        if text:
            return text
    except Exception:
        pass

    # deterministic fallback
    cur = pricing.get("currency","USD")
    subtotal = float(pricing.get("subtotal",0.0) or 0.0)
    tax = float(pricing.get("tax",0.0) or 0.0)
    total = float(pricing.get("total", subtotal + tax) or 0.0)
    include_tax = bool(pricing.get("include_tax", True))

    parts: list[str] = []
    parts.append(_build_status(call_meta))
    parts.append(f"Order: {(call_meta.get('order_type') or 'unspecified').title()}")
    name = (call_meta.get("customer_name") or "").strip()
    phone = (call_meta.get("phone") or "").strip()
    if name or phone:
        parts.append(f"Customer: {name or '—'}{' • ' + phone if phone else ''}")
    if call_meta.get("order_type") == "delivery":
        parts.append(f"Address: {call_meta.get('address') or '—'}")

    for ln in pricing.get("lines", []) or []:
        qty = int(ln.get("qty",1))
        desc = ln.get("desc","?")
        variant = (ln.get("variant_id") or "base")
        line_total = float(ln.get("line_total",0.0) or 0.0)
        item_str = f"- {qty}× {desc}"
        if variant and variant != "base":
            item_str += f" [{variant}]"
        item_str += f" — ${line_total:.2f}"
        attrs = ln.get("attributes") or {}
        bits=[]
        rice = (attrs.get("rice_type") or "").strip().lower()
        if rice in {"yellow","brown"}: bits.append(f"{rice} rice")
        spice=(attrs.get("spice_level") or "").strip().lower()
        if spice in {"mild","medium","spicy"}: bits.append(spice)
        if attrs.get("no_onions") is True: bits.append("no onions")
        if attrs.get("no_tomatoes") is True: bits.append("no tomatoes")
        if attrs.get("extra_white_sauce") is True: bits.append("extra white sauce")
        if attrs.get("extra_red_sauce") is True: bits.append("extra red sauce")
        style=(attrs.get("serving_style") or "").strip().lower()
        if style=="sauce_on_top": bits.append("sauce on top")
        elif style=="meat_on_side": bits.append("meat on the side")
        raw_notes = attrs.get("_notes")
        if isinstance(raw_notes, list): bits.extend([str(x) for x in raw_notes if x])
        elif isinstance(raw_notes, str) and raw_notes.strip(): bits.append(raw_notes.strip())
        if bits:
            seen=set(); ab=[]
            for s in bits:
                k=s.lower().strip()
                if k not in seen: ab.append(s); seen.add(k)
            item_str += " (" + ", ".join(ab) + ")"
        if float(ln.get("combo_total",0.0) or 0.0) > 0.0:
            item_str += " — Combo added"
        parts.append(item_str)

    parts.append(f"Subtotal: ${subtotal:.2f}")
    if include_tax: parts.append(f"Tax: ${tax:.2f}")
    parts.append(f"Total: ${total:.2f}")

    # note_bits = _notes_from_attributes(pricing.get("lines", []) or [])
    # note_bits.extend(_notes_from_transcript(transcript))
    note_bits = []
    for ln in (pricing.get("lines") or []):
        raw = (ln.get("attributes") or {}).get("_notes") or []
        if isinstance(raw, str):
            raw = [raw]
        for r in raw:
            s = " ".join(str(r).split())
            if s:
                note_bits.append(s)

    # dedupe preserving order
    seen = set();
    final = []
    for n in note_bits:
        k = n.lower()
        if k not in seen:
            final.append(n)
            seen.add(k)

    if final:
        parts.append("Notes: " + ", ".join(final))

    return "\n".join(parts)



# ---------- LLM phase 2: pretty summary ----------
# def _summarize(pricing: Dict[str, Any], collected: Dict[str, Any]) -> str:
#     sys = '''You are a precise restaurant agent. Produce a SHORT, human-friendly order recap from given Pricing JSON + transcript context.
#
# RENDERING RULES (apply in order)
# 1) One bullet per line in pricing.lines, in the same order.
#    - Format: "QTY× ITEM [variant if not base] — $LINE_TOTAL"
#    - If attributes exist, append in parentheses: "(yellow rice, spicy, meat on side)"
#    - If modifiers_detail exist, append " + <MOD_NAME> ×QTY ($AMT)" per modifier
#    - If combo_total > 0, append " — Combo added"
#    - If optimizer/bundle line, show description plainly and its line total (label it e.g., "Bundle" if obvious)
# 2) After items, show:
#    - "Subtotal: $X", and if pricing.include_tax, "Tax: $Y" and "Total: $Z".
# 3) NOTES section (only if any):
#    - Merge from:
#      a) attributes boolean flags like no_onions, no_tomatoes,
#      b) attributes._notes strings,
#      c) transcript hints (e.g., "singles only", "no deals") if present in the provided transcript.
#    - Render as: "Notes: <comma-separated>".
#
# STRICTNESS
# - Do NOT recompute math; use amounts from Pricing JSON as-is.
# - Do NOT expose internal keys or JSON.
# - Keep the whole summary within ~8 lines plus an optional single Notes line.
# - Append " — Combo added" ONLY if the line’s combo_total is a positive number in the provided JSON.
# - Do not infer combos from the transcript or attributes.
#
# If anything is missing or unclear, omit it rather than guessing.'''
#
#
#     user = (
#         "Create a concise human summary from Pricing JSON as-is (do not recompute math). "
#         "List each pricing line in order. If a line has `optimizer:true`, label it clearly as a bundle. "
#         "Show currency, Subtotal, Tax and Total. Keep it 6-8 lines max.\n\n"
#         f"Customer: {json.dumps(collected)}\n\n"
#         f"Pricing JSON:\n{json.dumps(pricing)}"
#     )
#
#     try:
#         rsp = client.chat.completions.create(
#             model=SUMMARY_MODEL,
#             temperature=0.3,
#             max_tokens=300,
#             messages=[
#                 {"role": "system", "content": sys},
#                 {"role": "user", "content": user},
#             ],
#         )
#         return (rsp.choices[0].message.content or "").strip()
#     except Exception as e:
#         # deterministic fallback
#         cur = pricing.get("currency", "USD")
#         subtotal = pricing.get("subtotal", 0.0)
#         tax = pricing.get("tax", 0.0)
#         total = pricing.get("total", 0.0)
#         parts = [f"Order summary (fallback): {cur} Subtotal {subtotal:.2f}, Tax {tax:.2f}, Total {total:.2f}."]
#         for i, ln in enumerate(pricing.get("lines", []), 1):
#             parts.append(f"{i}. {ln.get('desc','?')} x{ln.get('qty',1)} -> {ln.get('line_total',0):.2f}")
#         return "\n".join(parts)

# ---------- Endpoint ----------
@router.post("/ingest_v2")
def ingest_v2(payload: Dict[str, Any]):
    transcript: str = payload.get("transcript") or ""
    collected: Dict[str, Any] = payload.get("collected") or {}

    is_canceled = _detect_cancellation(transcript)
    needs_address = _needs_address(collected, transcript)
    call_meta = {
        "order_type": (collected.get("order_type") or "unspecified"),
        "customer_name": (collected.get("customer_name") or "").strip(),
        "phone": (collected.get("phone") or "").strip(),
        "address": (collected.get("address") or "").strip(),
        "status": "canceled" if is_canceled else "pending_address" if needs_address else "confirmed",
    }

    # Union menus → one extraction
    bundle = _load_menu_bundle(collected.get("menu_hint"))
    schema_ctx = _union_schema_from_bundle(bundle)

    extraction = _extract_cart(transcript, schema_ctx)
    cart_lines: List[Dict[str, Any]] = _repair_llm_lines(extraction.get("cart_lines") or [])
    line_models = _normalize_to_lines(cart_lines)

    if not line_models:
        summary = _summarize({}, collected, call_meta=call_meta, transcript=transcript)
        return {"ok": True, "has_order": False, "summary": summary, "pricing": None, "extraction": extraction}

    pricing = calc_core(
        payload_cart=line_models,
        payload_utterance=transcript,
        include_tax=payload.get("include_tax", True),
        menu_bundle=bundle,  # ← do this if you update calc_core/resolve_menu_for_line
    )

    summary = _summarize(pricing, collected, call_meta=call_meta, transcript=transcript)
    return {"ok": True, "has_order": True, "pricing": pricing, "summary": summary, "extraction": extraction}





@router.post("/ingest_eleven")
def ingest_eleven(payload: Dict[str, Any], request: Request):
    transcript_text = _flatten_eleven_transcript(payload, include_agent=True)

    customer_name = _extract_name(payload)
    phone = _get_caller_phone(payload)
    order_type = _detect_order_type(transcript_text)
    collected = {
        "customer_name": customer_name or "",
        "phone": phone or "",
        "order_type": order_type,
        "menu_hint": "middle-eastern",
    }

    is_canceled = _detect_cancellation(transcript_text)
    needs_address = _needs_address(collected, transcript_text)
    call_meta = {
        "order_type": (collected.get("order_type") or "unspecified"),
        "customer_name": (collected.get("customer_name") or "").strip(),
        "phone": (collected.get("phone") or "").strip(),
        "address": (collected.get("address") or "").strip(),
        "status": "canceled" if is_canceled else "pending_address" if needs_address else "confirmed",
    }

    # Union menus → one extraction
    bundle = _load_menu_bundle(collected.get("menu_hint"))
    schema_ctx = _union_schema_from_bundle(bundle)

    extraction = _extract_cart(transcript_text, schema_ctx)
    raw_lines: List[Dict[str, Any]] = _repair_llm_lines(extraction.get("cart_lines") or [])
    line_models = _normalize_to_lines(raw_lines)

    if not line_models:
        summary = _summarize({}, collected, call_meta=call_meta, transcript=transcript_text)
        if "text/plain" in (request.headers.get("accept","").lower()):
            return PlainTextResponse(summary)
        return {"ok": True, "has_order": False, "summary": summary, "pricing": None,
                "extraction": extraction, "collected": collected}

    pricing = calc_core(
        payload_cart=line_models,
        payload_utterance=transcript_text,
        include_tax=True,
        menu_bundle=bundle,  # ← if you wire it through
    )

    summary = _summarize(pricing, collected, call_meta=call_meta, transcript=transcript_text)
    if "text/plain" in (request.headers.get("accept","").lower()):
        return PlainTextResponse(summary)

    return {"ok": True, "has_order": True, "pricing": pricing, "summary": summary,
            "extraction": extraction, "collected": collected}
