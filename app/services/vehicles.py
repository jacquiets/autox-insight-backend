from typing import List, Dict, Any
from fastapi import HTTPException, status
from supabase import Client

async def get_brands_summary(supabase: Client) -> List[Dict[str, Any]]:
    """
    Obtiene el resumen estadístico de marcas y modelos consultando
    la vista materializada o estándar 'vw_brands_summary' en Supabase.
    """
    try:
        # La vista ya tiene la lógica del GROUP BY y la normalización de texto.
        # Ordenamos de forma descendente por la cantidad para cumplir con el requerimiento.
        response = (
            supabase.table("vw_brands_summary")
            .select("*")
            .order("cantidad_vehiculos", desc=True)
            .execute()
        )
        return response.data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al consultar el resumen estadístico de vehículos: {str(e)}"
        )
