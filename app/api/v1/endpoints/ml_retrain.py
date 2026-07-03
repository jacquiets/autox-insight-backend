"""
Reentrenamiento del modelo de IA (RF-15).

Ruta montada en: /api/v1/ml/retrain

Flujo:
  1. (Opcional) ejecuta el ETL para regenerar demanda_mensual.csv desde Supabase.
  2. Reentrena el XGBoost Regressor sobre los datos actualizados.
  3. Aplica el GATE de calidad (RNF-02): el nuevo model.pkl solo se promueve a
     producción si su wMAPE no degrada por encima del umbral establecido.
  4. Si se promueve, hace HOT-RELOAD del modelo en la RAM del proceso FastAPI
     sin necesidad de reiniciar el servidor.

Seguridad: requiere token de usuario autenticado (solo perfiles internos).
El reentrenamiento corre en un thread aparte para no bloquear el event-loop.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_current_user_token
from app.api.v1.endpoints.ml import reload_model

log = logging.getLogger(__name__)
router = APIRouter()

# Permite importar ml/train.py (vive fuera del paquete app/).
_ML_DIR = Path(__file__).resolve().parents[4] / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))


class RetrainRequest(BaseModel):
    correr_etl: bool = Field(
        False,
        description="Si es True, regenera demanda_mensual.csv desde Supabase antes de entrenar.",
    )
    forzar_promocion: bool = Field(
        False,
        description="Si es True, promueve el modelo aunque no pase el gate de wMAPE (usar con cuidado).",
    )


class RetrainResponse(BaseModel):
    promovido: bool = Field(..., description="True si el nuevo model.pkl reemplazó al vigente")
    forzado: bool
    version: str
    entrenado_en: str
    repuestos_conocidos: int
    modelo_recargado: bool = Field(..., description="True si el modelo se recargó en RAM tras promover")
    metrics: dict
    mensaje: str


def _run_etl_if_needed(correr_etl: bool) -> Optional[str]:
    if not correr_etl:
        return None
    try:
        import etl.etl_from_supabase as etl  # type: ignore
        etl.main()
        return "ETL ejecutado: demanda_mensual.csv regenerado desde Supabase."
    except Exception as e:  # noqa: BLE001
        log.error(f"Fallo el ETL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"El ETL falló antes del reentrenamiento: {e}",
        )


def _do_retrain(correr_etl: bool, forzar: bool) -> dict:
    """Trabajo pesado (bloqueante) — se ejecuta en un threadpool."""
    etl_msg = _run_etl_if_needed(correr_etl)

    import train  # ml/train.py  (via sys.path)
    result = train.run(force=forzar)

    result["_etl_msg"] = etl_msg
    return result


@router.post("/retrain", response_model=RetrainResponse, summary="Reentrenar el modelo IA (RF-15)")
async def retrain_model(
    req: RetrainRequest,
    token: str = Depends(get_current_user_token),
):
    """
    Reentrena el modelo con los datos mensuales más recientes y solo lo promueve
    a producción si supera el gate de calidad (wMAPE ≤ umbral). Tras promover,
    recarga el modelo en memoria en caliente (RF-15 + RNF-02).
    """
    try:
        result = await run_in_threadpool(_do_retrain, req.correr_etl, req.forzar_promocion)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("Error durante el reentrenamiento")
        raise HTTPException(status_code=500, detail=f"Error durante el reentrenamiento: {e}")

    promovido = bool(result.get("promoted"))
    recargado = False
    if promovido:
        reload_model()  # hot-reload en RAM (RF-15)
        recargado = True

    metrics = result.get("metrics", {})
    wmape = metrics.get("wmape")
    gate = metrics.get("wmape_gate")

    if promovido:
        mensaje = f"✅ Modelo promovido a producción (wMAPE {wmape}% ≤ gate {gate}%) y recargado en RAM."
    else:
        mensaje = (
            f"⛔ El nuevo modelo NO pasó el gate de calidad (wMAPE {wmape}% > gate {gate}%). "
            f"El modelo vigente se conserva intacto."
        )
    if result.get("_etl_msg"):
        mensaje = result["_etl_msg"] + " " + mensaje

    return RetrainResponse(
        promovido=promovido,
        forzado=bool(result.get("forced")),
        version=result.get("version", "?"),
        entrenado_en=result.get("trained_at", ""),
        repuestos_conocidos=int(result.get("known_parts", 0)),
        modelo_recargado=recargado,
        metrics=metrics,
        mensaje=mensaje,
    )
