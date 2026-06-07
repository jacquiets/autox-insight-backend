import math
from typing import Optional
from fastapi import APIRouter, Depends, Query
from supabase import Client

from app.core.supabase import get_supabase
from app.api.deps import get_current_user_token
from app.schemas.parts import PartListResponse, PartMetadata
from app.services.parts import get_paginated_parts

router = APIRouter()

@router.get("", response_model=PartListResponse, summary="Catálogo Maestro de Repuestos")
async def list_parts(
    page: int = Query(1, ge=1, description="Número de página para la paginación"),
    page_size: int = Query(10, ge=1, le=100, description="Cantidad de registros por página"),
    search: Optional[str] = Query(None, description="Búsqueda global (c_repuesto, descripcion, marca)"),
    supabase: Client = Depends(get_supabase),
    token: str = Depends(get_current_user_token),
) -> PartListResponse:
    """
    Retorna un listado paginado del catálogo de repuestos con soporte para
    búsqueda global (ILIKE) en los campos principales.
    """
    
    # 1. Obtener la data filtrada y el conteo total del servicio
    data, total_records = await get_paginated_parts(
        supabase=supabase,
        page=page,
        page_size=page_size,
        search=search,
    )

    # 2. Calcular matemática de la paginación
    total_pages = math.ceil(total_records / page_size) if total_records > 0 else 1

    metadata = PartMetadata(
        total_records=total_records,
        current_page=page,
        page_size=page_size,
        total_pages=total_pages
    )

    # 3. Retornar JSON (200 OK implícito en FastAPI)
    return PartListResponse(
        metadata=metadata,
        data=data
    )
