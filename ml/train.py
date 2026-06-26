"""
Entrena el modelo XGBoost de predicción de demanda y guarda ml/model.pkl.

Ejecutar DESPUÉS de etl/etl_from_supabase.py:
    python ml/train.py

Features: [codigo_enc, mes, anio, km_enc, tipo_enc]
Target:   cantidad_total (unidades demandadas en el mes)
"""
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data" / "clean" / "demanda_mensual.csv"
MODEL_OUT = Path(__file__).parent / "model.pkl"

FEATURE_COLS = ["codigo_enc", "mes", "anio", "km_enc"]
TARGET_COL = "cantidad_total"


def load_data() -> pd.DataFrame:
    if not CLEAN.exists():
        raise FileNotFoundError(
            f"No se encontró {CLEAN}.\n"
            "Ejecuta primero: python etl/etl_from_supabase.py"
        )
    df = pd.read_csv(CLEAN)
    log.info(f"Dataset cargado: {len(df)} filas, {df['producto_id'].nunique()} repuestos únicos")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # ── Encoders categóricos ────────────────────────────────────────────────
    repuesto_map = {v: i for i, v in enumerate(sorted(df["producto_id"].unique()))}

    df["codigo_enc"] = df["producto_id"].map(repuesto_map)

    # ── Normalizar km (escala logarítmica para reducir outliers) ────────────
    df["km_enc"] = np.log1p(pd.to_numeric(df["km_promedio"], errors="coerce").fillna(0))

    # ── Target ──────────────────────────────────────────────────────────────
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Descartar filas incompletas
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df[TARGET_COL] > 0]  # descartar demanda 0 (no aporta al modelo)

    log.info(f"Filas válidas para entrenamiento: {len(df)}")
    log.info(f"  Target — min: {df[TARGET_COL].min():.1f} | "
             f"media: {df[TARGET_COL].mean():.2f} | max: {df[TARGET_COL].max():.1f}")

    encoder = {
        "repuesto_map": repuesto_map,
        "anio_default": int(df["anio"].mode()[0]),
        "km_default": float(df["km_promedio"].median()),
    }
    return df[FEATURE_COLS + [TARGET_COL]], encoder


def train(df: pd.DataFrame) -> XGBRegressor:
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    n = len(df)
    # Con pocos datos usamos más del dataset para entrenamiento
    test_size = 0.15 if n < 300 else 0.20

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Hiperparámetros calibrados para datasets pequeños (~150-400 filas)
    model = XGBRegressor(
        n_estimators=150,
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

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Métricas ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    log.info(f"📈 MAE en test:  {mae:.4f} unidades")
    log.info(f"📈 MAPE en test: {mape:.2f}%")

    # Cross-validation (3 folds para datasets pequeños)
    cv_scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    log.info(f"📈 MAE cross-val (3-fold): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_))
    log.info("📊 Feature importance:")
    for feat, imp in sorted(feat_imp.items(), key=lambda x: -x[1]):
        log.info(f"   {feat:15s}: {imp:.4f}")

    return model


if __name__ == "__main__":
    df_raw = load_data()
    df_feat, encoder = build_features(df_raw)
    model = train(df_feat)

    bundle = {
        "model": model,
        "encoder": encoder,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "version": "2.0",
    }

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)

    log.info(f"✅ Modelo guardado en {MODEL_OUT}")
    log.info("   Siguiente paso: copiar ml/model.pkl → autox-insight-backend/ml/model.pkl")
