from fastapi import APIRouter

api_router = APIRouter()

# Placeholders for future endpoint routers:
# from app.api.v1.endpoints import users, items
# api_router.include_router(users.router, prefix="/users", tags=["users"])
# api_router.include_router(items.router, prefix="/items", tags=["items"])

@api_router.get("/health", tags=["health"])
def health_check():
    """
    Simple health check endpoint to verify API and routing functionality.
    """
    return {"status": "ok", "message": "API V1 is functioning correctly"}
