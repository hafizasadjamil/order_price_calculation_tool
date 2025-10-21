import re
from functools import lru_cache
from typing import Dict, Any, List
from fastapi import HTTPException
from app.db.mongo import menus
from app.utils.common import TENANT_HINT_RE, ME_KEYS, US_KEYS

def parse_menu_key_from_utterance(u: str) -> Dict[str, str]:
    u = (u or "").strip().lower()
    if not u:
        return {}
    m = re.search(r"(?:menu|store|tenant)\s*:\s*([a-z0-9\-_]{2,64})", u)
    if m:
        return {"menu_slug": m.group(1)}
    if any(k in u for k in US_KEYS):
        return {"menu_slug": "american"}
    if any(k in u for k in ME_KEYS):
        return {"menu_slug": "middle-eastern"}
    m2 = TENANT_HINT_RE.search(u)
    if m2:
        slug = re.sub(r"\s+", "-", m2.group(1).strip())
        return {"menu_slug": slug}
    return {}

def load_default_menu() -> Dict[str, Any]:
    doc = menus.find_one({"meta.is_default": True}, {"_id": 0})
    if not doc:
        doc = menus.find_one({}, {"_id": 0}) or {}
    return doc or {}

@lru_cache(maxsize=256)
def load_menu_by_slug(menu_slug: str) -> Dict[str, Any]:
    doc = menus.find_one({"meta.menu_slug": menu_slug}, {"_id": 0})
    if not doc:
        doc = menus.find_one({"menu_slug": menu_slug}, {"_id": 0})
    if not doc:
        raise KeyError(f"Menu not found for slug '{menu_slug}'")
    return doc

@lru_cache(maxsize=1)
def _load_all_menus() -> List[Dict[str, Any]]:
    try:
        return list(menus.find({}, {"_id": 0})) or []
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

@lru_cache(maxsize=1)
def _catalog_item_to_menu_slug() -> Dict[str, str]:
    docs = _load_all_menus()
    mapping = {}
    for doc in docs:
        slug = ((doc.get("meta") or {}).get("menu_slug") or doc.get("menu_slug") or "")
        for cat in doc.get("categories", []):
            for it in cat.get("items", []):
                iid = it.get("id")
                if iid and slug:
                    mapping[iid] = slug
    return mapping

@lru_cache(maxsize=256)
def _menu_by_slug_cached(slug: str) -> Dict[str, Any]:
    try:
        return load_menu_by_slug(slug)
    except Exception:
        for doc in _load_all_menus():
            s = (doc.get("meta", {}) or {}).get("menu_slug") or doc.get("menu_slug")
            if not s:
                nm = (doc.get("name") or doc.get("meta", {}).get("restaurant") or "menu").lower()
                s = re.sub(r"[^a-z0-9]+", "-", nm).strip("-")
            if s == slug:
                return doc
        raise

def resolve_menu_for_line(ln, payload_utterance: str) -> Dict[str, Any]:
    if ln.menu_hint:
        return _menu_by_slug_cached(ln.menu_hint)

    u = (ln.utterance or payload_utterance or "").lower()
    if any(k in u for k in US_KEYS):
        try: return _menu_by_slug_cached("american")
        except Exception: pass
    if any(k in u for k in ME_KEYS):
        try: return _menu_by_slug_cached("middle-eastern")
        except Exception: pass

    slug = _catalog_item_to_menu_slug().get(ln.item_id)
    if slug:
        return _menu_by_slug_cached(slug)

    return load_default_menu()
