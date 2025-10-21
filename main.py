from fastapi import FastAPI
from pydantic import BaseModel, validator
from typing import List, Union
import re

app = FastAPI()

def money(n: float) -> float:
    return round((n + 1e-12) * 100) / 100

class SimplePayload(BaseModel):

    prices: Union[str, List[float]]

    @validator("prices", pre=True)
    def parse_prices(cls, v):

        if isinstance(v, str):
            nums = re.findall(r"[\d]+(?:\.\d+)?", v)
            return [float(x) for x in nums]
        return v

@app.post("/calc-simple")
def calc_simple(payload: SimplePayload):

    TAX_RATE = 0.06

    subtotal = money(sum(payload.prices))
    tax = money(subtotal * TAX_RATE)
    total = money(subtotal + tax)

    return {
        "subtotal": subtotal,
        "tax": tax,
        "total": total
    }
