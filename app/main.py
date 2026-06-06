from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.DEBUG else None,
    docs_url=f"{settings.API_V1_STR}/docs" if settings.DEBUG else None,
    redoc_url=f"{settings.API_V1_STR}/redoc" if settings.DEBUG else None,
)

# Set all CORS enabled origins
# In production, specify actual allowed origins instead of wildcard "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["root"])
def root_route():
    """
    Root endpoint redirecting or displaying basic API info.
    """
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs_url": f"{settings.API_V1_STR}/docs" if settings.DEBUG else None,
        "status": "active"
    }
