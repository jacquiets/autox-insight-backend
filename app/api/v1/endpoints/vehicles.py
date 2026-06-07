from typing import List
from fastapi import APIRouter, Depends
from supabase import Client

from app.core.supabase import get_supabase
from app.api.deps import get_current_user_token
from app.schemas.vehicles import VehicleBrandSummary
from app.services.vehicles import get_brands_summary

router = APIRouter()

@router.get("/brands-summary", response_model=List[VehicleBrandSummary], summary="Resumen estadístico de marcas y modelos")
async def get_brands_summary_endpoint(
    supabase: Client = Depends(get_supabase),
    token: str = Depends(get_current_user_token),
) -> List[VehicleBrandSummary]:
    """
    Retorna una lista agrupada de marcas y modelos, con la cantidad
    total de vehículos por combinación. La lógica de agrupación corre
    del lado de PostgreSQL (Supabase) a través de una vista por eficiencia.
    """
    summary_data = await get_brands_summary(supabase)
    return summary_data
