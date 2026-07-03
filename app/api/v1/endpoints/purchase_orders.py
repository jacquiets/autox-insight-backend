"""
Órdenes de Compra Inteligentes (RF-12).

Ruta montada en: /api/v1/purchase-orders/*

  • GET  /purchase-orders/suggestions  → propuestas de OC basadas en IA (stock vs demanda predicha)
  • POST /purchase-orders/generate     → persiste una OC inteligente en orden_compra_detalle

Cierra el ciclo predicción → acción logística: el mismo motor XGBoost que predice
la demanda es el que dispara las sugerencias de reposición (Objetivo de Negocio 5).
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from supabase import Client

from app.core.supabase import get_supabase
from app.api.deps import get_current_user_token
from app.services.purchase_orders import (
    build_smart_purchase_proposals,
    persist_purchase_order,
)

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PurchaseProposal(BaseModel):
    codigo_repuesto: str
    descripcion: Optional[str] = None
    marca: Optional[str] = None
    stock_actual: float
    stock_minimo: float
    stock_maximo: float
    demanda_ia: float = Field(..., description="Demanda proyectada por el modelo XGBoost")
    deficit: float = Field(..., description="max(0, demanda_ia − stock_actual)")
    compra_sugerida: float
    confianza_ia: float
    etiqueta_confianza: str
    repuesto_conocido: bool
    en_quiebre: bool


class SuggestionsResponse(BaseModel):
    resumen: dict
    propuestas: List[PurchaseProposal]


class GenerateItem(BaseModel):
    codigo_repuesto: str
    compra_sugerida: float = Field(..., gt=0)


class GenerateRequest(BaseModel):
    items: List[GenerateItem] = Field(..., min_length=1)
    observacion: Optional[str] = None


class GenerateResponse(BaseModel):
    n_oc: str
    items_insertados: int
    detalle: list
    mensaje: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/suggestions", response_model=SuggestionsResponse, summary="Sugerencias de OC por IA (RF-12)")
async def get_suggestions(
    mes: int = Query(..., ge=1, le=12, description="Mes objetivo de la predicción"),
    anio: Optional[int] = Query(None, description="Año objetivo (opcional)"),
    km: Optional[float] = Query(None, ge=0, description="Kilometraje promedio de la flota"),
    solo_quiebres: bool = Query(True, description="Solo repuestos en riesgo de quiebre"),
    limite: int = Query(50, ge=1, le=200, description="Máximo de propuestas a devolver"),
    supabase: Client = Depends(get_supabase),
    token: str = Depends(get_current_user_token),
):
    """
    Genera propuestas de reposición cruzando el stock físico con la demanda
    proyectada por el motor de IA. Prioriza los repuestos con mayor déficit.
    """
    return await build_smart_purchase_proposals(
        supabase=supabase,
        mes=mes,
        anio=anio,
        km=km,
        solo_quiebres=solo_quiebres,
        limite=limite,
    )


@router.post("/generate", response_model=GenerateResponse, summary="Generar OC inteligente (RF-12)")
async def generate_purchase_order(
    req: GenerateRequest,
    supabase: Client = Depends(get_supabase),
    token: str = Depends(get_current_user_token),
):
    """
    Persiste una Orden de Compra inteligente en `orden_compra_detalle`, marcando
    su origen como IA para trazabilidad (tipo_requerimiento_compra = 'IA_PREDICTIVO').
    """
    return await persist_purchase_order(
        supabase=supabase,
        items=[item.model_dump() for item in req.items],
        observacion=req.observacion,
    )
