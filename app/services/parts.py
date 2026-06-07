from typing import Tuple, List, Dict, Any, Optional
from fastapi import HTTPException, status
from supabase import Client

async def get_paginated_parts(
    supabase: Client,
    page: int,
    page_size: int,
    search: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    
    # Preparamos la consulta pidiendo el count exacto de registros
    query = supabase.table("repuesto").select("c_repuesto, descripcion, marca", count="exact")

    # Filtro global ILIKE (case-insensitive) sobre código, descripción y marca
    if search:
        search_term = f"%{search}%"
        or_filter = (
            f"c_repuesto.ilike.{search_term},"
            f"descripcion.ilike.{search_term},"
            f"marca.ilike.{search_term}"
        )
        query = query.or_(or_filter)

    # Lógica de paginación usando LIMIT y OFFSET embebido en .range()
    start = (page - 1) * page_size
    end = start + page_size - 1
    query = query.range(start, end)
    
    # Orden default por código de repuesto
    query = query.order("c_repuesto", desc=False)

    try:
        response = query.execute()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al consultar el catálogo de repuestos: {str(e)}"
        )
        
    total_count = response.count if response.count is not None else 0
    return response.data, total_count
