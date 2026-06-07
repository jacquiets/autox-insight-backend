import math
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from supabase import Client

from app.core.supabase import get_supabase
from app.api.deps import get_current_user_token
from app.schemas.work_orders import WorkOrderListResponse, WorkOrderMetadata, WorkOrderPart
from app.services.work_orders import get_paginated_work_orders, get_work_order_parts

router = APIRouter()

@router.get("", response_model=WorkOrderListResponse, summary="Listar Órdenes de Trabajo")
async def list_work_orders(
    page: int = Query(1, ge=1, description="Número de página para la paginación"),
    page_size: int = Query(10, ge=1, le=100, description="Cantidad de registros por página"),
    search: Optional[str] = Query(None, description="Búsqueda global (n_ot, placa, vin, requerimiento)"),
    c_estado: Optional[str] = Query(None, description="Filtro exacto por código de estado"),
    marca: Optional[str] = Query(None, description="Filtro exacto por marca de vehículo"),
    tipo_ot_desc: Optional[str] = Query(None, description="Filtro parcial por tipo de orden"),
    fecha_inicio: Optional[date] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    fecha_fin: Optional[date] = Query(None, description="Fecha de fin (YYYY-MM-DD)"),
    supabase: Client = Depends(get_supabase),
    token: str = Depends(get_current_user_token),
) -> WorkOrderListResponse:
    """
    Retorna un listado paginado y filtrado de órdenes de trabajo, 
    incluyendo la información embebida del vehículo (auto) y su estado.
    """
    
    # Llamar a la capa de servicio
    data, total_records = await get_paginated_work_orders(
        supabase=supabase,
        page=page,
        page_size=page_size,
        search=search,
        c_estado=c_estado,
        marca=marca,
        tipo_ot_desc=tipo_ot_desc,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
    )

    # Calcular paginación
    total_pages = math.ceil(total_records / page_size) if total_records > 0 else 1

    # Construir metadata
    metadata = WorkOrderMetadata(
        total_records=total_records,
        current_page=page,
        page_size=page_size,
        total_pages=total_pages
    )

    # En este punto `data` es una lista de diccionarios que Pydantic
    # parseará automáticamente al formato anidado de WorkOrderItem
    return WorkOrderListResponse(
        metadata=metadata,
        data=data
    )


@router.get("/{n_ot}/parts", response_model=List[WorkOrderPart], summary="Repuestos de una Orden de Trabajo")
async def list_work_order_parts(
    n_ot: str,
    supabase: Client = Depends(get_supabase),
    token: str = Depends(get_current_user_token),
) -> List[WorkOrderPart]:
    """
    Retorna la lista de repuestos consumidos en una orden de trabajo específica.
    Si la orden no tiene repuestos o no existe, retorna una lista vacía.
    """
    parts = await get_work_order_parts(supabase, n_ot)
    return parts
