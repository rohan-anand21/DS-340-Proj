from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routers import predictions, pages

app = FastAPI(title="Multi-Model Hotel Sentiment Analysis Prediction")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(pages.router)
app.include_router(
    predictions.router,
    prefix="/api",
    tags=["predictions"]
)