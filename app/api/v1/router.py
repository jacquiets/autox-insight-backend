from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth, work_orders, ml, ml_retrain, vehicles, parts, purchase_orders,
)

api_router = APIRouter()

# ── Endpoints registrados ─────────────────────────────────────────────────────
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(work_orders.router, prefix="/work-orders", tags=["work-orders"])
api_router.include_router(ml.router, prefix="/ml", tags=["prediccion-ml"])
api_router.include_router(ml_retrain.router, prefix="/ml", tags=["prediccion-ml"])
api_router.include_router(vehicles.router, prefix="/vehicles", tags=["vehicles"])
api_router.include_router(parts.router, prefix="/parts", tags=["parts"])
api_router.include_router(purchase_orders.router, prefix="/purchase-orders", tags=["ordenes-compra-ia"])

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
