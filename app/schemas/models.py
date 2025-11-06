# from typing import List, Optional, Dict, Any, Union
# from pydantic import BaseModel, Field, validator
# from app.utils.common import _coerce_cart_list
#
# #models.py
#
#
# class Mod(BaseModel):
#     modifier_id: str
#     qty: int = 1
#     unit_price: Optional[float] = None
#
# class Line(BaseModel):
#     item_id: str
#     qty: int
#     variant_id: Optional[str] = None
#     modifiers: List[Mod] = Field(default_factory=list)
#     combo_opt_in: bool = False
#     attributes: Optional[Dict[str, Any]] = None
#     utterance: Optional[str] = None
#     menu_hint: Optional[str] = None  # e.g., "american" or "middle-eastern"
#
# class Payload(BaseModel):
#     cart: Union[str, List[Line], List[Any]]
#     utterance: Optional[str] = ""
#
#     @validator("cart", pre=True)
#     def v_cart(cls, v):
#         if isinstance(v, str) and "cart_lines_with" in v:
#             return []
#         return _coerce_cart_list(v)
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from app.utils.common import _coerce_cart_list

# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------

class Mod(BaseModel):
    """Modifier for a menu item (e.g. extra chicken, no onions, etc.)"""
    modifier_id: str
    qty: int = 1
    unit_price: Optional[float] = None


class Line(BaseModel):
    """Single item in the cart (main course, appetizer, drink, etc.)"""
    item_id: str
    qty: int
    variant_id: Optional[str] = None
    modifiers: List[Mod] = Field(default_factory=list)
    combo_opt_in: bool = False
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict)
    utterance: Optional[str] = None
    menu_hint: Optional[str] = None  # e.g., "american" or "middle-eastern"

    class Config:
        extra = "ignore"  # ignore unexpected keys from LLM to prevent crashes


class Payload(BaseModel):
    """Cart payload passed to calc_core for pricing."""
    cart: Union[str, List[Line], List[Any]]
    utterance: Optional[str] = ""

    @validator("cart", pre=True)
    def v_cart(cls, v):
        # Defensive guard for malformed strings
        if isinstance(v, str) and "cart_lines_with" in v:
            return []
        return _coerce_cart_list(v)


# ---------------------------------------------------------------------
# OPTIONAL (NEW) — for clarity when you postcall
# ---------------------------------------------------------------------

class PostCallInput(BaseModel):
    """Request body schema for /postcall/ingest."""
    transcript: str
    collected: Dict[str, Any] = Field(default_factory=dict)
    call: Dict[str, Any] = Field(default_factory=dict)
    include_tax: bool = True


class PostCallOutput(BaseModel):
    """Response schema for /postcall/ingest."""
    ok: bool
    has_order: bool
    summary_text: str
    pricing: Optional[Dict[str, Any]] = None
    extraction: Dict[str, Any] = Field(default_factory=dict)
