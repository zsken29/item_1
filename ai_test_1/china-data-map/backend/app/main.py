"""
China City Data Map - FastAPI Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .database import engine, Base
from .router import cities_router, stats_router
from .config import settings

# Create database directory
os.makedirs("./data", exist_ok=True)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="China City Data Map API",
    description="中国城市人均数据地图 API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cities_router)
app.include_router(stats_router)


@app.get("/")
def root():
    return {"message": "China City Data Map API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
