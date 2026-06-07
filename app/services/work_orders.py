from datetime import date
from typing import Tuple, List, Dict, Any, Optional
from fastapi import HTTPException, status
from supabase import Client


async def get_paginated_work_orders(
    supabase: Client,
    page: int,
    page_size: int,
    search: Optional[str] = None,
    c_estado: Optional[str] = None,
    marca: Optional[str] = None,
    tipo_ot_desc: Optional[str] = None,
    fecha_inicio: Optional[date] = None,
    fecha_fin: Optional[date] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Construye y ejecuta la consulta a Supabase con paginación, filtros relacionales 
    y búsqueda global.
    """
    
    # Query de selección. Usamos alias "estado" y "auto".
    # Usamos !inner en auto para poder filtrar eficientemente a nivel de join si es necesario.
    select_query = (
        "n_ot, fecha, km, rango_km, c_estado, tipo_ot_desc, requerimiento, dias_atencion, "
        "estado:estado_ot(c_estado, descripcion), "
        "auto:auto!inner(vin, placa, marca, modelo, anio_mod, tipo_transmision, tipo_combustible)"
    )

    query = supabase.table("orden_trabajo").select(select_query, count="exact")

    # 1. Filtros exactos y condicionales simples
    if c_estado:
        query = query.eq("c_estado", c_estado)
        
    if tipo_ot_desc:
        query = query.ilike("tipo_ot_desc", f"%{tipo_ot_desc}%")
        
    if fecha_inicio:
        query = query.gte("fecha", fecha_inicio.isoformat())
        
    if fecha_fin:
        query = query.lte("fecha", fecha_fin.isoformat())
        
    if marca:
        # Filtro en la tabla foránea (PostgREST soporta foreign table filtering con !inner)
        query = query.eq("auto.marca", marca)

    # 2. Búsqueda Global (Global Search)
    if search:
        search_term = f"%{search}%"
        
        # Debido a que PostgREST no soporta ORs cruzando diferentes tablas fácilmente,
        # consultamos primero si el término hace match con alguna 'placa'.
        try:
            auto_resp = supabase.table("auto").select("vin").ilike("placa", search_term).execute()
        except Exception:
            auto_resp = None
            
        or_conditions = [
            f"n_ot.ilike.{search_term}",
            f"vin.ilike.{search_term}",
            f"requerimiento.ilike.{search_term}"
        ]
        
        # Si encontró autos con esa placa, agregamos la condición vin.in.(vin1,vin2...)
        if auto_resp and auto_resp.data:
            vins_matching = [item["vin"] for item in auto_resp.data]
            vins_csv = ",".join(vins_matching)
            or_conditions.append(f"vin.in.({vins_csv})")
            
        query = query.or_(",".join(or_conditions))

    # 3. Paginación eficiente mediante OFFSET y LIMIT (en Supabase es a través de .range inclusivo)
    start = (page - 1) * page_size
    end = start + page_size - 1
    query = query.range(start, end)
    
    # 4. Ordenamiento por defecto
    query = query.order("fecha", desc=True, nullsfirst=False)

    # 5. Ejecutar consulta
    try:
        response = query.execute()
    except Exception as e:
        # En caso de error de base de datos o sintaxis (ej. timeout o error de cast)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al consultar la base de datos: {str(e)}"
        )
        
    # count retorna el conteo sin el range(start, end) aplicado
    total_count = response.count if response.count is not None else 0
    return response.data, total_count


async def get_work_order_parts(
    supabase: Client,
    n_ot: str
) -> List[Dict[str, Any]]:
    """
    Obtiene los repuestos de una orden de trabajo específica y hace un JOIN
    con la tabla maestra `repuesto` para extraer la marca.
    """
    # Query de selección: relacionamos con 'repuesto' a través de producto_id = c_repuesto
    select_query = (
        "id, producto_id, descripcion, cantidad, precio_unitario, total, c_moneda, "
        "repuesto:repuesto(marca)"
    )
    
    try:
        response = (
            supabase.table("ot_repuesto")
            .select(select_query)
            .eq("n_ot", n_ot)
            .order("id")
            .execute()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al consultar los repuestos de la OT: {str(e)}"
        )
        
    # Mapear la respuesta para "aplanar" el campo marca
    # PostgREST retorna `{"repuesto": {"marca": "Toyota"}}`, nosotros lo pasaremos al nivel superior
    parts = []
    for row in response.data:
        marca_val = None
        if row.get("repuesto") and isinstance(row["repuesto"], dict):
            marca_val = row["repuesto"].get("marca")
            
        part_data = {
            "id": row["id"],
            "producto_id": row.get("producto_id"),
            "descripcion": row.get("descripcion"),
            "marca": marca_val,
            "cantidad": row.get("cantidad"),
            "precio_unitario": row.get("precio_unitario"),
            "total": row.get("total"),
            "c_moneda": row.get("c_moneda"),
        }
        parts.append(part_data)
        
    # Si la lista está vacía (porque la OT no existe o no tiene repuestos),
    # simplemente se retorna un array vacío [], que FastAPI devolverá como 200 OK.
    return parts
