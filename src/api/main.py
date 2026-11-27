"""FastAPI entry: routers, static UI, model warmup."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import get_dynamic_bundle, get_static_model
from api.routers import predict
from static_inference import STATIC_MODEL_PATH

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm model caches at startup."""
    try:
        get_static_model()
        print(f"Loaded static classifier: {STATIC_MODEL_PATH}")
    except Exception as exc:
        print(f"Warning: static model not loaded yet ({exc})")
    try:
        bundle = get_dynamic_bundle()
        print(
            f"Loaded dynamic J/Z model: method={bundle.get('method')} "
            f"mode={bundle.get('mode')}"
        )
    except Exception as exc:
        print(f"Warning: dynamic model not loaded yet ({exc})")
    yield


app = FastAPI(
    title="SignScript",
    description="ASL letter demo (landmarks only): static + J/Z motion.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predict.router)


@app.get("/")
def index():
    """Serve the browser demo page."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html missing")
    return FileResponse(index_path)


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
