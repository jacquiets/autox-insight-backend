"""
Servicio de Órdenes de Compra Inteligentes (RF-12).

Cruza el stock físico (tabla `stock`) contra la demanda proyectada por el
motor de IA (XGBoost) para detectar repuestos con riesgo de quiebre y proponer
cantidades óptimas de reposición. Opcionalmente persiste las propuestas en
`orden_compra_detalle`, cerrando el ciclo predicción → acción logística.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from supabase import Client

from app.api.v1.endpoints.ml import PredictRequest, _predict_one, _bundle

log = logging.getLogger(__name__)


def _predict_demanda(codigo: str, mes: int, anio: Optional[int], km: Optional[float]) -> Dict[str, Any]:
    """Invoca el motor de IA para un SKU y normaliza su salida."""
    if not _bundle:
        # Sin modelo cargado, degradamos: demanda 0 y confianza nula.
        return {"cantidad": 0.0, "confianza": 0.0, "etiqueta": "Modelo no disponible",
                "conocido": False}
    pred = _predict_one(PredictRequest(codigo_repuesto=codigo, mes=mes, anio=anio, km=km))
    return {
        "cantidad": pred.cantidad_estimada,
        "confianza": pred.confianza,
        "etiqueta": pred.etiqueta_confianza,
        "conocido": pred.repuesto_conocido,
    }


async def build_smart_purchase_proposals(
    supabase: Client,
    mes: int,
    anio: Optional[int],
    km: Optional[float],
    solo_quiebres: bool,
    limite: int,
) -> Dict[str, Any]:
    """
    Genera propuestas de OC inteligentes:
      déficit = max(0, demanda_predicha_IA − stock_actual)
      compra_sugerida = déficit ajustado al stock_maximo cuando aplica.
    """
    # 1. Traer stock + datos del repuesto en un solo query relacional.
    select_q = (
        "c_repuesto, stock, stock_minimo, stock_maximo, "
        "repuesto:repuesto(descripcion, marca)"
    )
    try:
        resp = supabase.table("stock").select(select_q).execute()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al consultar stock: {e}",
        )

    proposals: List[Dict[str, Any]] = []
    for row in resp.data or []:
        codigo = row.get("c_repuesto")
        if not codigo:
            continue

        stock_actual = float(row.get("stock") or 0)
        stock_min = float(row.get("stock_minimo") or 0)
        stock_max = float(row.get("stock_maximo") or 0)
        rep = row.get("repuesto") or {}
        descripcion = rep.get("descripcion") if isinstance(rep, dict) else None
        marca = rep.get("marca") if isinstance(rep, dict) else None

        pred = _predict_demanda(codigo, mes, anio, km)
        demanda_ia = pred["cantidad"]

        # Déficit proyectado: lo que la IA dice que se consumirá menos lo disponible.
        deficit = max(0.0, round(demanda_ia - stock_actual, 2))

        # Cantidad a comprar: cubrir el déficit y, si hay stock_max, apuntar a él.
        if stock_max > 0:
            objetivo = max(stock_max, demanda_ia)
            compra_sugerida = max(0.0, round(objetivo - stock_actual, 2))
        else:
            compra_sugerida = deficit

        en_quiebre = stock_actual < demanda_ia or (stock_min > 0 and stock_actual <= stock_min)

        if solo_quiebres and not en_quiebre:
            continue
        if compra_sugerida <= 0 and solo_quiebres:
            continue

        proposals.append({
            "codigo_repuesto": codigo,
            "descripcion": descripcion,
            "marca": marca,
            "stock_actual": stock_actual,
            "stock_minimo": stock_min,
            "stock_maximo": stock_max,
            "demanda_ia": demanda_ia,
            "deficit": deficit,
            "compra_sugerida": compra_sugerida,
            "confianza_ia": pred["confianza"],
            "etiqueta_confianza": pred["etiqueta"],
            "repuesto_conocido": pred["conocido"],
            "en_quiebre": en_quiebre,
        })

    # Priorizar por mayor déficit (mayor riesgo primero).
    proposals.sort(key=lambda p: p["deficit"], reverse=True)
    proposals = proposals[:limite]

    resumen = {
        "total_propuestas": len(proposals),
        "en_quiebre": sum(1 for p in proposals if p["en_quiebre"]),
        "unidades_a_comprar": round(sum(p["compra_sugerida"] for p in proposals), 2),
        "mes": mes,
        "anio": anio,
    }
    return {"resumen": resumen, "propuestas": proposals}


async def persist_purchase_order(
    supabase: Client,
    items: List[Dict[str, Any]],
    observacion: Optional[str],
) -> Dict[str, Any]:
    """
    Persiste una OC inteligente en `orden_compra_detalle`.
    Marca el origen como generada por IA para trazabilidad.
    """
    if not items:
        raise HTTPException(status_code=400, detail="No hay ítems para generar la orden de compra.")

    n_oc = f"OC-IA-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    obs = observacion or "Orden de Compra generada automáticamente por el motor de IA (XGBoost demand-forecast)."

    rows = []
    for it in items:
        cantidad = float(it.get("compra_sugerida") or it.get("cantidad_compra") or 0)
        if cantidad <= 0:
            continue
        rows.append({
            "n_oc": n_oc,
            "repuesto_id": it["codigo_repuesto"],
            "cantidad_compra": cantidad,
            "observacion": obs,
            "tipo_requerimiento_compra": "IA_PREDICTIVO",
        })

    if not rows:
        raise HTTPException(status_code=400, detail="Ningún ítem tiene cantidad de compra válida (> 0).")

    try:
        resp = supabase.table("orden_compra_detalle").insert(rows).execute()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al persistir la orden de compra: {e}",
        )

    return {
        "n_oc": n_oc,
        "items_insertados": len(rows),
        "detalle": resp.data,
        "mensaje": f"Orden de Compra {n_oc} generada con {len(rows)} ítems (origen: IA).",
    }
