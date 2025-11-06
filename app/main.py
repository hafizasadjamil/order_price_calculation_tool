from fastapi import FastAPI
from app.router import calc, tools, postcall_smart as rest_ingest_v2
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Voice Agent Pricing", version="2.1")

app.include_router(calc.router, prefix="", tags=["calc"])
app.include_router(tools.router, tags=["tools"])
app.include_router(rest_ingest_v2.router)  # exposes POST /rest/ingest_v2
