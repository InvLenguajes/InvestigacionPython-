from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import recommendations
from app.db.connection import test_connection
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Movie Recommender ML API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "app": "Movie Recommender ML API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Root"])
def health_check():
    db_ok = test_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "environment": settings.app_env,
    }