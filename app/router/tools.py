# tools_router.py (or app/router/tools.py)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
from app.schemas.models import Line
from app.services.pricing_engine import calc_core
#tool file contain endpoints
router = APIRouter()

# ---------- Request Schemas ----------
class CartPatch(BaseModel):
    lines: List[Line]

class InterimTotalReq(BaseModel):
    cart_patch: CartPatch
    utterance: Optional[str] = ""

class FinalizeReq(BaseModel):
    cart_patch: CartPatch
    utterance: Optional[str] = ""
    include_tax: bool = True



@router.post("/tool/interim_total")
def interim_total(req: InterimTotalReq):
    """
    Stateless subtotal calculator for active cart updates (no tax yet).
    """
    if not req.cart_patch or not req.cart_patch.lines:
        raise HTTPException(status_code=400, detail="cart_patch.lines is required")

    result = calc_core(req.cart_patch.lines, req.utterance or "", include_tax=False)
    result["utterance"] = req.utterance or ""
    result["note"] = "Interim: Subtotal only (stateless)"
    return result


# ---------- FINALIZE ----------
@router.post("/tool/finalize")
def finalize(req: FinalizeReq):
    """
    Stateless final calculator (includes tax).
    """
    if not req.cart_patch or not req.cart_patch.lines:
        raise HTTPException(status_code=400, detail="cart_patch.lines is required")

    final_lines = req.cart_patch.lines
    result = calc_core(final_lines, req.utterance or "", include_tax=req.include_tax)

    result["utterance"] = req.utterance or ""
    result["note"] = "Final: Tax + Total "
    return result



@router.post("/webhook/summary")
async def receive_summary(request: Request):
    """
    Receives a POST webhook from n8n with a JSON body like:
    { "summary": "text summary here" }
    Prints it neatly to the console.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    print("\n" + "="*60)
    print("📦  New Order Summary Received")
    print("="*60)
    print(json.dumps(body, indent=2))

    # Extract the summary field (Option B)
    summary = body.get("summary")

    print("\n📝 Summary Text:")
    print(summary if summary else "[No summary provided]")
    print("="*60 + "\n")

    return {"received": True, "summary_received": bool(summary)}