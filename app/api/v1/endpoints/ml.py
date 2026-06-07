"""
Endpoint de prediccion de demanda de repuestos (ML).

Ruta montada en: /api/v1/ml/predict
"""
import pickle
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter()

# ── Carga del modelo en memoria ───────────────────────────────────────────────
# model.pkl vive en /ml/model.pkl relativo a la raiz del proyecto
_MODEL_PATH = Path(__file__).parent.parent.parent.parent.parent / "ml" / "model.pkl"
_bundle: dict = {}


def load_model() -> None:
    """Carga el modelo ML. Llamar desde el lifespan del app o al inicio."""
    global _bundle
    if not _MODEL_PATH.exists():
        log.warning(f"model.pkl no encontrado en {_MODEL_PATH}. POST /ml/predict retornara 503.")
        return
    with open(_MODEL_PATH, "rb") as f:
        _bundle = pickle.load(f)
    encoder = _bundle.get("encoder", {})
    log.info(
        f"Modelo ML cargado (v{_bundle.get('version', '?')}) | "
        f"{len(encoder.get('repuesto_map', {}))} repuestos conocidos"
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    codigo_repuesto: str = Field(
        ...,
        description="Codigo exacto del repuesto en Supabase (campo producto_id / c_repuesto)",
        examples=["FILTRO-01", "ACC-223"]
    )
    mes: int = Field(..., ge=1, le=12, description="Mes objetivo de la prediccion (1-12)")
    anio: Optional[int] = Field(
        None,
        description="Anio objetivo. Si se omite, usa el anio mas frecuente del entrenamiento."
    )
    km: Optional[float] = Field(
        None,
        ge=0,
        description="Kilometraje promedio del vehiculo. Si se omite, usa la mediana historica."
    )


class PredictResponse(BaseModel):
    codigo_repuesto: str
    mes: int
    anio: int
    cantidad_estimada: float = Field(..., description="Unidades estimadas a demandar en ese mes")
    confianza: float = Field(..., ge=0.0, le=1.0, description="Nivel de confianza 0-1")
    repuesto_conocido: bool = Field(
        ...,
        description="True si el repuesto fue visto durante el entrenamiento"
    )
    mae_referencia: float = Field(
        default=4.33,
        description="MAE de referencia del modelo (margen de error promedio en unidades)"
    )


class MLStatusResponse(BaseModel):
    modelo_cargado: bool
    version: Optional[str] = None
    repuestos_conocidos: Optional[int] = None
    mae_referencia: float = 4.33
    features: list[str] = ["codigo_enc", "mes", "anio", "km_enc"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _predict_one(req: PredictRequest) -> PredictResponse:
    encoder = _bundle.get("encoder", {})
    repuesto_map: dict = encoder.get("repuesto_map", {})

    codigo_enc = repuesto_map.get(req.codigo_repuesto, -1)
    repuesto_conocido = req.codigo_repuesto in repuesto_map
    anio = req.anio if req.anio is not None else encoder.get("anio_default", 2025)
    km = req.km if req.km is not None else encoder.get("km_default", 0.0)
    km_enc = math.log1p(km)

    features = np.array([[codigo_enc, req.mes, anio, km_enc]], dtype=float)
    raw = float(_bundle["model"].predict(features)[0])
    cantidad = max(0.0, round(raw, 2))
    confianza = 0.87 if repuesto_conocido else 0.45

    return PredictResponse(
        codigo_repuesto=req.codigo_repuesto,
        mes=req.mes,
        anio=anio,
        cantidad_estimada=cantidad,
        confianza=confianza,
        repuesto_conocido=repuesto_conocido,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", response_model=MLStatusResponse, summary="Estado del modulo ML")
def ml_status():
    """Retorna el estado del modelo cargado en memoria."""
    if not _bundle:
        return MLStatusResponse(modelo_cargado=False)
    encoder = _bundle.get("encoder", {})
    return MLStatusResponse(
        modelo_cargado=True,
        version=_bundle.get("version"),
        repuestos_conocidos=len(encoder.get("repuesto_map", {})),
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predecir demanda de un repuesto",
)
def predict(req: PredictRequest):
    """
    Predice la cantidad estimada de unidades demandadas de un repuesto para un mes/anio dado.

    - Confianza **0.87** si el repuesto fue visto en entrenamiento.
    - Confianza **0.45** si es un repuesto nuevo (extrapolacion).
    - El campo `mae_referencia` indica el margen de error promedio del modelo (~4.33 unidades).
    """
    if not _bundle:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo ML no disponible. "
                "Verifica que ml/model.pkl existe en el servidor (Railway)."
            ),
        )
    return _predict_one(req)
