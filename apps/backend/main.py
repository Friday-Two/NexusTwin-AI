from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.backend.routes.assistant import router as assistant_router
from apps.backend.routes.auth import router as auth_router
from apps.backend.routes.executions import router as executions_router
from apps.backend.routes.negotiations import router as negotiations_router
from apps.backend.routes.preferences import router as preferences_router
from apps.backend.routes.services import router as services_router
from apps.backend.routes.users import router as users_router
from database.connection import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="NexusTwin AI",
    description="Privacy-aware autonomous multi-agent digital twin platform.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(assistant_router)
app.include_router(users_router)
app.include_router(preferences_router)
app.include_router(services_router)
app.include_router(negotiations_router)
app.include_router(executions_router)


@app.get("/")
def root():
    return {
        "name": "NexusTwin AI",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "NexusTwin AI",
    }
