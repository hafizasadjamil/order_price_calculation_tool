from fastapi import FastAPI
from app.router import calc, tools

app = FastAPI(title="Voice Agent Pricing", version="2.1")
app.include_router(calc.router, prefix="", tags=["calc"])
app.include_router(tools.router, tags=["tools"])
