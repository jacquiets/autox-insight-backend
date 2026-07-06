"""
Motor de predicción de demanda de repuestos (IA / Machine Learning).

Modelo: demand-forecast v3 · XGBoost Regressor
Ruta montada en: /api/v1/ml/*

Endpoints:
  • GET  /ml/status   → estado del modelo + métricas de salud (RF-11)
  • POST /ml/predict  → predicción puntual con confianza REAL calculada (RF-09, RF-10)

── SOBRE LA CONFIANZA (RF-10) ────────────────────────────────────────────────
La confianza YA NO es un valor fijo. Se calcula en runtime combinando:
  1. Densidad histórica del SKU  → ¿cuántos meses de historia tiene el repuesto?
  2. Magnitud de la demanda       → los SKUs de alta rotación se predicen con
     mucho menor error relativo (ver métricas del entrenamiento).
Un repuesto se etiqueta "Alta Confiabilidad" cuando supera el umbral del 80%
definido en los objetivos del proyecto; de lo contrario se advierte al usuario
que la predicción es una extrapolación de baja confianza.
"""
import pickle
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter()

# ── Carga del modelo en memoria ───────────────────────────────────────────────
# model.pkl vive en ml/model.pkl relativo a la raíz del repo.
#   __file__ = app/api/v1/endpoints/ml.py  →  subir 5 niveles → raíz → /ml/model.pkl
_MODEL_PATH = Path(__file__).resolve().parents[4] / "ml" / "model.pkl"
_bundle: dict = {}

# Umbral de negocio: predicción "Alta Confiabilidad" (definido en los objetivos).
CONFIDENCE_HIGH_THRESHOLD = 0.80


def load_model() -> None:
    """Carga el modelo ML en RAM. Llamar desde el lifespan del app."""
    global _bundle
    if not _MODEL_PATH.exists():
        log.info(f"model.pkl no encontrado en {_MODEL_PATH}. Intentando descargar desde Supabase Storage...")
        from app.core.config import settings
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            try:
                _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                resp = httpx.get(
                    f"{settings.SUPABASE_URL}/storage/v1/object/modelos-ia/model-v4.pkl",
                    headers={
                        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
                        "apikey": settings.SUPABASE_KEY
                    },
                    timeout=30
                )
                if resp.status_code != 200:
                    resp = httpx.get(
                        f"{settings.SUPABASE_URL}/storage/v1/object/modelos-ia/model.pkl",
                        headers={
                            "Authorization": f"Bearer {settings.SUPABASE_KEY}",
                            "apikey": settings.SUPABASE_KEY
                        },
                        timeout=30
                    )
                resp.raise_for_status()
                with open(_MODEL_PATH, "wb") as f:
                    f.write(resp.content)
                log.info(f"Modelo descargado desde Supabase Storage ({len(resp.content)} bytes)")
            except Exception as e:
                log.warning(f"No se pudo descargar modelo desde Storage: {e}")

    if not _MODEL_PATH.exists():
        log.warning(f"model.pkl no encontrado en {_MODEL_PATH}. Endpoints /ml/* degradados a 503.")
        _bundle = {}
        return
    with open(_MODEL_PATH, "rb") as f:
        _bundle = pickle.load(f)
    encoder = _bundle.get("encoder", {})
    metrics = _bundle.get("metrics", {})
    log.info(
        f"Modelo ML cargado (demand-forecast v{_bundle.get('version', '?')}) | "
        f"{len(encoder.get('repuesto_map', {}))} repuestos conocidos | "
        f"wMAPE={metrics.get('wmape', '?')}%"
    )


def reload_model() -> None:
    """Recarga el modelo desde disco (hot-reload tras un reentrenamiento — RF-15)."""
    load_model()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    codigo_repuesto: str = Field(
        ...,
        description="Código exacto del repuesto en Supabase (producto_id / c_repuesto)",
        examples=["01001-01001", "FILTRO-01"],
    )
    mes: int = Field(..., ge=1, le=12, description="Mes objetivo de la predicción (1-12)")
    anio: Optional[int] = Field(
        None, description="Año objetivo. Si se omite, usa el año más frecuente del entrenamiento."
    )
    km: Optional[float] = Field(
        None, ge=0,
        description="Kilometraje promedio del vehículo. Si se omite, usa la mediana histórica.",
    )


class PredictResponse(BaseModel):
    codigo_repuesto: str
    mes: int
    anio: int
    cantidad_estimada: float = Field(..., description="Unidades estimadas a demandar en ese mes")
    confianza: float = Field(..., ge=0.0, le=1.0, description="Nivel de confianza calculado (0-1)")
    alta_confiabilidad: bool = Field(
        ..., description=f"True si confianza ≥ {CONFIDENCE_HIGH_THRESHOLD:.0%} (umbral de negocio)"
    )
    etiqueta_confianza: str = Field(
        ..., description="'Alta Confiabilidad' | 'Confianza Media' | 'Extrapolación (baja confianza)'"
    )
    repuesto_conocido: bool = Field(
        ..., description="True si el repuesto fue visto durante el entrenamiento"
    )
    observaciones_historicas: int = Field(
        ..., description="Nº de meses de historia que tiene este SKU (base de la confianza)"
    )
    mae_referencia: float = Field(..., description="MAE del modelo (margen de error promedio en unidades)")
    explicacion: str = Field(..., description="Explicación legible de la confianza para el usuario")


class MetricsBlock(BaseModel):
    mae: Optional[float] = None
    mape_global: Optional[float] = None
    wmape: Optional[float] = None
    mape_alta_rotacion: Optional[float] = None
    mae_alta_rotacion: Optional[float] = None
    wmape_gate: Optional[float] = None
    feature_importance: Optional[dict] = None


class MLStatusResponse(BaseModel):
    modelo_cargado: bool
    algoritmo: str = "XGBoost Regressor"
    modelo: str = "demand-forecast"
    version: Optional[str] = None
    repuestos_conocidos: Optional[int] = None
    entrenado_en: Optional[str] = None
    features: list[str] = []
    umbral_alta_confiabilidad: float = CONFIDENCE_HIGH_THRESHOLD
    metrics: Optional[MetricsBlock] = None


# ── Cálculo de confianza REAL (RF-10) ─────────────────────────────────────────

def _compute_confidence(codigo: str, cantidad: float) -> tuple[float, int]:
    """
    Confianza calculada combinando densidad histórica del SKU y magnitud de demanda.

    Devuelve (confianza 0-1, observaciones_historicas).

    Lógica:
      • Sin historia (repuesto nuevo)        → 0.40 (extrapolación pura).
      • Historia escasa (< min_obs)          → 0.55–0.70 según nº de observaciones.
      • Historia sólida + demanda alta        → hasta 0.92 (el error relativo es bajo).
      • Historia sólida + demanda muy baja    → penalizada (el % de error se dispara).
    """
    encoder = _bundle.get("encoder", {})
    profile: dict = encoder.get("sku_profile", {})
    min_obs = encoder.get("confidence_min_obs", 4)
    high_rot = encoder.get("high_rotation_min", 5)

    info = profile.get(codigo)
    if info is None:
        return 0.40, 0  # repuesto nuevo → extrapolación

    obs = int(info.get("obs_count", 0))
    demanda_med = float(info.get("demanda_med", 1.0))

    # Componente 1: densidad histórica (satura hacia 1 con más observaciones).
    #   obs= min_obs → ~0.6 ; obs=12 → ~0.85 ; obs>=24 → ~0.95
    dens = 1.0 - math.exp(-obs / max(min_obs, 1) * 0.9)

    # Componente 2: magnitud de demanda (los SKUs de alta rotación se predicen mejor).
    #   demanda_med >= high_rotation → boost ; demanda_med muy baja → penalización.
    mag = min(1.0, (demanda_med / (high_rot * 2.0)) + 0.35)

    confianza = 0.35 + 0.65 * (0.6 * dens + 0.4 * mag)
    confianza = round(max(0.0, min(0.95, confianza)), 2)
    return confianza, obs


def _label_for(confianza: float) -> str:
    if confianza >= CONFIDENCE_HIGH_THRESHOLD:
        return "Alta Confiabilidad"
    if confianza >= 0.60:
        return "Confianza Media"
    return "Extrapolación (baja confianza)"


def _explain(confianza: float, obs: int, conocido: bool) -> str:
    if not conocido:
        return (
            "Repuesto NUEVO para el modelo: la predicción es una extrapolación. "
            "Se recomienda validación manual del Jefe de Almacén."
        )
    if confianza >= CONFIDENCE_HIGH_THRESHOLD:
        return (
            f"SKU con {obs} meses de historia y rotación estable: el modelo predice "
            f"con alta confiabilidad (≥ {CONFIDENCE_HIGH_THRESHOLD:.0%})."
        )
    if confianza >= 0.60:
        return (
            f"SKU con {obs} meses de historia: confianza media. La demanda de baja "
            f"magnitud eleva el error relativo; usar como referencia."
        )
    return (
        f"SKU con historia limitada ({obs} meses) o demanda muy esporádica: "
        f"confianza baja, tratar la cifra como orientativa."
    )


# ── Inferencia ────────────────────────────────────────────────────────────────

def _predict_one(req: PredictRequest) -> PredictResponse:
    encoder = _bundle.get("encoder", {})
    metrics = _bundle.get("metrics", {})
    repuesto_map: dict = encoder.get("repuesto_map", {})
    feature_cols: list = _bundle.get("feature_cols", ["codigo_enc", "mes", "anio", "km_enc"])

    codigo_enc = repuesto_map.get(req.codigo_repuesto, -1)
    repuesto_conocido = req.codigo_repuesto in repuesto_map
    anio = req.anio if req.anio is not None else encoder.get("anio_default", 2025)
    km = req.km if req.km is not None else encoder.get("km_default", 0.0)
    km_enc = math.log1p(km)
    tipo_enc = encoder.get("tipo_default", 0)

    # Rotación histórica del SKU (feature más importante del modelo v3).
    profile = encoder.get("sku_profile", {})
    rotacion_enc = float(profile.get(req.codigo_repuesto, {}).get("obs_count",
                                     encoder.get("rotacion_default", 1)))

    # Construir el vector respetando el orden EXACTO de feature_cols del bundle.
    feature_values = {
        "codigo_enc": codigo_enc,
        "mes": req.mes,
        "anio": anio,
        "km_enc": km_enc,
        "tipo_enc": tipo_enc,
        "rotacion_enc": rotacion_enc,
    }
    features = np.array([[feature_values.get(c, 0) for c in feature_cols]], dtype=float)

    raw = float(_bundle["model"].predict(features)[0])
    cantidad = max(0.0, round(raw, 2))

    confianza, obs = _compute_confidence(req.codigo_repuesto, cantidad)
    etiqueta = _label_for(confianza)

    return PredictResponse(
        codigo_repuesto=req.codigo_repuesto,
        mes=req.mes,
        anio=anio,
        cantidad_estimada=cantidad,
        confianza=confianza,
        alta_confiabilidad=confianza >= CONFIDENCE_HIGH_THRESHOLD,
        etiqueta_confianza=etiqueta,
        repuesto_conocido=repuesto_conocido,
        observaciones_historicas=obs,
        mae_referencia=metrics.get("mae", 4.33),
        explicacion=_explain(confianza, obs, repuesto_conocido),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", response_model=MLStatusResponse, summary="Estado y salud del modelo IA (RF-11)")
def ml_status():
    """
    Retorna el estado del modelo cargado en RAM junto con sus métricas de salud
    (wMAPE, MAPE de alta rotación, MAE, importancia de features).
    """
    if not _bundle:
        return MLStatusResponse(modelo_cargado=False)
    encoder = _bundle.get("encoder", {})
    metrics = _bundle.get("metrics", {})
    return MLStatusResponse(
        modelo_cargado=True,
        version=_bundle.get("version"),
        repuestos_conocidos=len(encoder.get("repuesto_map", {})),
        entrenado_en=_bundle.get("trained_at"),
        features=_bundle.get("feature_cols", []),
        metrics=MetricsBlock(**{k: metrics.get(k) for k in MetricsBlock.model_fields}),
    )


@router.post("/predict", response_model=PredictResponse, summary="Predecir demanda con confianza real (RF-09/RF-10)")
def predict(req: PredictRequest):
    """
    Predice la cantidad estimada de un repuesto para un mes/año dado, usando el
    **kilometraje** como variable de contexto (RF-09), y devuelve un nivel de
    **confianza calculado en runtime** (RF-10) con su etiqueta de confiabilidad.
    """
    if not _bundle:
        raise HTTPException(
            status_code=503,
            detail="Modelo ML no disponible. Verifica que ml/model.pkl existe en el servidor (Railway).",
        )
    return _predict_one(req)
