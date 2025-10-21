from fastapi import FastAPI
from app.router import calc, tools

def create_app() -> FastAPI:
    app = FastAPI(title="Voice Agent Pricing", version="2.1")
    app.include_router(calc.router, prefix="", tags=["calc"])
    app.include_router(tools.router, tags=["tools"])
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
