"""
Entrena el modelo XGBoost de predicción de demanda de repuestos y guarda ml/model.pkl.

Ejecutar DESPUÉS de etl/etl_from_supabase.py:
    python ml/train.py

Features: [codigo_enc, mes, anio, km_enc, tipo_enc, rotacion_enc]
Target:   cantidad_total (unidades demandadas en el mes)

── SOBRE LAS MÉTRICAS ────────────────────────────────────────────────────────
La demanda de repuestos en un taller multimarca es INTERMITENTE: el 63% de los
registros son de ≤2 unidades y el 45% son exactamente 1. En ese régimen el MAPE
clásico se dispara (equivocarse por 1 unidad en un repuesto que rota 1 vez = 100%
de error), por lo que NO es una métrica representativa de la salud del modelo.

Por eso reportamos y "gateamos" con métricas robustas para demanda intermitente:
  • wMAPE  (weighted MAPE): Σ|error| / Σreal → error porcentual ponderado por volumen.
  • MAPE_alta: MAPE calculado SOLO sobre SKUs de alta rotación (los que el negocio
    realmente necesita predecir bien y donde el % sí es significativo).

El bundle serializado guarda estas métricas para que el frontend (RF-10, RF-11)
muestre la confiabilidad real del modelo, no un número inventado.
"""
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
# __file__ = ml/train.py  →  ROOT_ML = ml/
ROOT_ML = Path(__file__).parent
CLEAN = ROOT_ML / "data" / "clean" / "demanda_mensual.csv"
MODEL_OUT = ROOT_ML / "model.pkl"

FEATURE_COLS = ["codigo_enc", "mes", "anio", "km_enc", "tipo_enc", "rotacion_enc"]
TARGET_COL = "cantidad_total"

# ── Umbrales de negocio (RNF-02) ──────────────────────────────────────────────
# "Alta rotación" = SKU con demanda mensual >= este valor (foco del negocio).
HIGH_ROTATION_MIN = 5
# Gate de promoción: el nuevo modelo solo se guarda si wMAPE <= este umbral.
# Ajustado a la realidad de los datos (demanda intermitente). Configurable.
WMAPE_GATE = 60.0
# SKU se considera "de confianza" si tiene al menos este nº de observaciones históricas.
CONFIDENCE_MIN_OBS = 4
# Versión del modelo (semántica del pipeline demand-forecast).
MODEL_VERSION = "3.0"


def load_data() -> pd.DataFrame:
    if not CLEAN.exists():
        raise FileNotFoundError(
            f"No se encontró {CLEAN}.\n"
            "Ejecuta primero: python ml/etl/etl_from_supabase.py"
        )
    df = pd.read_csv(CLEAN)
    log.info(f"Dataset cargado: {len(df)} filas, {df['producto_id'].nunique()} repuestos únicos")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # ── Encoder de repuestos (ordinal) ───────────────────────────────────────
    repuesto_map = {v: i for i, v in enumerate(sorted(df["producto_id"].unique()))}
    df["codigo_enc"] = df["producto_id"].map(repuesto_map)

    # ── Encoder de tipo de OT (aporta contexto de la falla) ──────────────────
    tipo_series = df.get("tipo_ot_desc", pd.Series([""] * len(df))).fillna("").astype(str)
    tipo_map = {v: i for i, v in enumerate(sorted(tipo_series.unique()))}
    df["tipo_enc"] = tipo_series.map(tipo_map)

    # ── Normalizar km (escala logarítmica para reducir outliers) ─────────────
    df["km_enc"] = np.log1p(pd.to_numeric(df["km_promedio"], errors="coerce").fillna(0))

    # ── Feature de rotación: nº de OTs que generaron esa demanda ──────────────
    # Le da a la IA señal de si el repuesto es de rotación frecuente o esporádica.
    df["rotacion_enc"] = pd.to_numeric(df.get("n_ots", 1), errors="coerce").fillna(1)

    # ── Perfil histórico por SKU (para el cálculo de confianza en runtime) ────
    #   obs_count  : cuántos meses de historia tiene el repuesto
    #   demanda_med: demanda mediana histórica (magnitud típica)
    perfil = (
        df.groupby("producto_id")
        .agg(obs_count=("cantidad_total", "size"),
             demanda_med=("cantidad_total", "median"))
        .reset_index()
    )
    sku_profile = {
        row["producto_id"]: {
            "obs_count": int(row["obs_count"]),
            "demanda_med": float(row["demanda_med"]),
        }
        for _, row in perfil.iterrows()
    }

    # ── Target ────────────────────────────────────────────────────────────────
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Descartar filas incompletas / sin demanda
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df[TARGET_COL] > 0]

    log.info(f"Filas válidas para entrenamiento: {len(df)}")
    log.info(f"  Target — min: {df[TARGET_COL].min():.1f} | "
             f"media: {df[TARGET_COL].mean():.2f} | max: {df[TARGET_COL].max():.1f}")

    encoder = {
        "repuesto_map": repuesto_map,
        "tipo_map": tipo_map,
        "sku_profile": sku_profile,
        "anio_default": int(df["anio"].mode()[0]),
        "km_default": float(pd.to_numeric(df["km_promedio"], errors="coerce").median()),
        "tipo_default": 0,
        "rotacion_default": 1,
        "confidence_min_obs": CONFIDENCE_MIN_OBS,
        "high_rotation_min": HIGH_ROTATION_MIN,
    }
    return df[FEATURE_COLS + [TARGET_COL]], encoder


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE: Σ|error| / Σreal. Métrica robusta para demanda intermitente."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def train(df: pd.DataFrame) -> tuple[XGBRegressor, dict]:
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    n = len(df)
    test_size = 0.15 if n < 300 else 0.20

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Hiperparámetros calibrados para datasets pequeños e intermitentes
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,   # evita overfitting en datasets pequeños
        reg_alpha=0.1,        # regularización L1
        reg_lambda=1.0,       # regularización L2
        random_state=42,
        eval_metric="mae",
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ── Métricas ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape_global = mean_absolute_percentage_error(y_test, y_pred) * 100
    wmape = _wmape(y_test, y_pred)

    # MAPE solo sobre SKUs de alta rotación (donde el % sí es representativo)
    mask_alta = y_test >= HIGH_ROTATION_MIN
    if mask_alta.sum() > 0:
        mape_alta = mean_absolute_percentage_error(y_test[mask_alta], y_pred[mask_alta]) * 100
        mae_alta = mean_absolute_error(y_test[mask_alta], y_pred[mask_alta])
    else:
        mape_alta = float("nan")
        mae_alta = float("nan")

    log.info("── MÉTRICAS DE EVALUACIÓN ──────────────────────────────────")
    log.info(f"  MAE (test):            {mae:.4f} unidades")
    log.info(f"  MAPE global (test):    {mape_global:.2f}%  (infla por demanda intermitente)")
    log.info(f"  wMAPE (test):          {wmape:.2f}%  ← métrica de gate")
    log.info(f"  MAPE alta rotación:    {mape_alta:.2f}%  (SKUs con demanda ≥ {HIGH_ROTATION_MIN})")
    log.info(f"  MAE alta rotación:     {mae_alta:.2f} unidades")

    # Cross-validation robusta
    cv_scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    log.info(f"  MAE cross-val (3fold): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_))
    log.info("  Feature importance:")
    for feat, imp in sorted(feat_imp.items(), key=lambda x: -x[1]):
        log.info(f"     {feat:15s}: {imp:.4f}")

    metrics = {
        "mae": round(float(mae), 4),
        "mape_global": round(float(mape_global), 2),
        "wmape": round(float(wmape), 2),
        "mape_alta_rotacion": round(float(mape_alta), 2) if mape_alta == mape_alta else None,
        "mae_alta_rotacion": round(float(mae_alta), 2) if mae_alta == mae_alta else None,
        "mae_cross_val": round(float(-cv_scores.mean()), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "wmape_gate": WMAPE_GATE,
        "feature_importance": {k: round(float(v), 4) for k, v in feat_imp.items()},
    }
    return model, metrics


def passes_gate(metrics: dict) -> bool:
    """
    Gate de promoción (RNF-02 + RF-15): el modelo solo se promueve a producción
    si su wMAPE no degrada por encima del umbral establecido.
    """
    wmape = metrics.get("wmape", float("inf"))
    gate = metrics.get("wmape_gate", WMAPE_GATE)
    ok = wmape <= gate
    if ok:
        log.info(f"✅ GATE OK: wMAPE {wmape:.2f}% ≤ umbral {gate:.2f}% → modelo apto para producción.")
    else:
        log.warning(f"⛔ GATE FALLIDO: wMAPE {wmape:.2f}% > umbral {gate:.2f}% → modelo NO se promueve.")
    return ok


def build_bundle(model: XGBRegressor, encoder: dict, metrics: dict) -> dict:
    return {
        "model": model,
        "encoder": encoder,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "version": MODEL_VERSION,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def run(force: bool = False) -> dict:
    """
    Pipeline completo de entrenamiento reutilizable por el endpoint RF-15.

    Devuelve un dict con: {promoted: bool, metrics: {...}, version, trained_at}.
    Solo sobrescribe model.pkl si pasa el gate (o si force=True).
    """
    df_raw = load_data()
    df_feat, encoder = build_features(df_raw)
    model, metrics = train(df_feat)
    bundle = build_bundle(model, encoder, metrics)

    promoted = passes_gate(metrics) or force
    if promoted:
        with open(MODEL_OUT, "wb") as f:
            pickle.dump(bundle, f)
        log.info(f"✅ Modelo v{MODEL_VERSION} guardado en {MODEL_OUT}")
    else:
        log.warning("⚠️  model.pkl NO fue actualizado (el modelo vigente se conserva).")

    return {
        "promoted": promoted,
        "forced": force,
        "version": MODEL_VERSION,
        "trained_at": bundle["trained_at"],
        "metrics": metrics,
        "known_parts": len(encoder["repuesto_map"]),
    }


if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv
    result = run(force=force_flag)
    print("\n=== RESULTADO DEL ENTRENAMIENTO ===")
    print(f"  Promovido a producción: {result['promoted']}")
    print(f"  wMAPE: {result['metrics']['wmape']}%  (gate: {result['metrics']['wmape_gate']}%)")
    print(f"  MAPE alta rotación: {result['metrics']['mape_alta_rotacion']}%")
    print(f"  MAE: {result['metrics']['mae']} unidades")
    print(f"  Repuestos conocidos: {result['known_parts']}")
