from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from app.utils.common import _coerce_cart_list

class Mod(BaseModel):
    modifier_id: str
    qty: int = 1
    unit_price: Optional[float] = None

class Line(BaseModel):
    item_id: str
    qty: int
    variant_id: Optional[str] = None
    modifiers: List[Mod] = Field(default_factory=list)
    combo_opt_in: bool = False
    attributes: Optional[Dict[str, Any]] = None
    utterance: Optional[str] = None
    menu_hint: Optional[str] = None  # e.g., "american" or "middle-eastern"

class Payload(BaseModel):
    cart: Union[str, List[Line], List[Any]]
    utterance: Optional[str] = ""

    @validator("cart", pre=True)
    def v_cart(cls, v):
        if isinstance(v, str) and "cart_lines_with" in v:
            return []
        return _coerce_cart_list(v)
