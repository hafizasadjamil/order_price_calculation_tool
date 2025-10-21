from fastapi import APIRouter
from typing import Dict, Any
from app.schemas.models import Payload, Line
from app.services.pricing_engine import calc_core

router = APIRouter()

@router.post("/calc")
def calc(payload: Payload) -> Dict[str, Any]:
    """
    Mixed-menu calculator (back-compat):
    - resolves menu per line (menu_hint → keywords → item catalog → default)
    - applies pricing/combos/modifiers per menu
    - runs pack optimizer across eligible appetizer lines
    """
    payload_u = payload.utterance or ""
    # Normalize incoming list to Line models
    cart_lines = []
    for ln_dict in (payload.cart or []):
        ln = Line(**ln_dict) if not isinstance(ln_dict, Line) else ln_dict
        cart_lines.append(ln)

    return calc_core(cart_lines, payload_u)
