from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
import time
from pydantic import BaseModel

from app.schemas.models import Line
from app.services.pricing_engine import calc_core

router = APIRouter()


SESSIONS: Dict[str, Dict[str, Any]] = {}

def now_ts() -> int:
    return int(time.time())

def _key_for(l: Line) -> tuple:
    return (l.item_id, l.variant_id or "base")

def merge_cart(existing: List[Line], patch: List[Line]) -> List[Line]:
    index: Dict[tuple, int] = {}
    for i, l in enumerate(existing):
        index[_key_for(l)] = i
    for p in patch:
        k = _key_for(p)
        if k in index:
            i = index[k]
            existing[i].qty = max(0, existing[i].qty + p.qty)
            if p.modifiers: existing[i].modifiers = p.modifiers
            if p.attributes: existing[i].attributes = p.attributes
            if p.menu_hint: existing[i].menu_hint = p.menu_hint
            if p.utterance: existing[i].utterance = p.utterance
        else:
            existing.append(p)
    return [l for l in existing if l.qty > 0]

# ---- Request models ----
class CartPatch(BaseModel):
    lines: List[Line]

class InterimTotalReq(BaseModel):
    order_id: str
    currency: str = "USD"
    tax_included: bool = False  # kept for backward compatibility (ignored here)
    cart_patch: Optional[CartPatch] = None
    utterance: Optional[str] = ""

class FinalizeReq(BaseModel):
    order_id: str
    currency: str = "USD"
    tax_included: bool = False

# ---- Endpoints ----

@router.post("/tool/interim_total")
def interim_total(req: InterimTotalReq):
    """
    Deferred Tax: Interim shows Subtotal only (no tax).
    User can still add more items later; we re-price every time.
    """
    sess = SESSIONS.setdefault(req.order_id, {"lines": [], "finalized": False, "history": []})
    if sess.get("finalized"):
        raise HTTPException(status_code=409, detail="Order already finalized")

    if req.cart_patch and req.cart_patch.lines:
        sess["lines"] = merge_cart(sess["lines"], req.cart_patch.lines)

    # include_tax=False for interim
    result = calc_core(sess["lines"], req.utterance or "", include_tax=False)
    # helpful UI cue:
    result["note"] = "Interim: Subtotal only (tax deferred)"

    sess["history"].append({"ts": now_ts(), "type": "interim", "calc": result})
    return result


@router.post("/tool/finalize")
def finalize(req: FinalizeReq):
    """
    Deferred Tax: Final call must add Tax + Total exactly once.
    """
    sess = SESSIONS.get(req.order_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown order_id")

    # include_tax=True for final
    result = calc_core(sess["lines"], "", include_tax=True)
    result["note"] = "Final: Tax + Total applied once"

    sess["history"].append({"ts": now_ts(), "type": "final", "calc": result})
    sess["finalized"] = True
    return result
