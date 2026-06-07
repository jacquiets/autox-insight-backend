from fastapi import APIRouter

from app.api.v1.endpoints import auth, ml

api_router = APIRouter()

# ── Endpoints registrados ─────────────────────────────────────────────────────
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(ml.router, prefix="/ml", tags=["prediccion-ml"])

# Agrega aquí futuros routers:
# from app.api.v1.endpoints import users, orders
# api_router.include_router(users.router, prefix="/users", tags=["users"])
# api_router.include_router(orders.router, prefix="/orders", tags=["orders"])


@api_router.get("/health", tags=["health"])
def health_check():
    """
    Simple health check endpoint to verify API and routing functionality.
    """
    return {"status": "ok", "message": "API V1 is functioning correctly"}
