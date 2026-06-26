"""
ETL: Extrae datos de demanda desde Supabase y genera el CSV de entrenamiento.

Uso:
    python etl/etl_from_supabase.py

Genera:
    data/clean/demanda_mensual.csv
"""
import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_CSV = CLEAN_DIR / "demanda_mensual.csv"
RAW_CSV = ROOT / "data" / "raw" / "demanda_raw.csv"


def get_supabase_client() -> Client:
    load_dotenv(ROOT / ".env")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL y SUPABASE_SERVICE_KEY son requeridos en .env")
    return create_client(url, key)


def fetch_all(client: Client, table: str, select: str = "*", page_size: int = 1000) -> list[dict]:
    """Paginación automática para tablas grandes."""
    records = []
    offset = 0
    while True:
        resp = client.table(table).select(select).range(offset, offset + page_size - 1).execute()
        batch = resp.data
        if not batch:
            break
        records.extend(batch)
        log.info(f"  [{table}] Descargados {len(records)} registros...")
        if len(batch) < page_size:
            break
        offset += page_size
    return records


def extract(client: Client) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extrae las tres tablas necesarias desde Supabase."""
    log.info("Extrayendo orden_trabajo...")
    ot_records = fetch_all(
        client, "orden_trabajo",
        select="n_ot,vin,fecha,anio,mes,km,tipo_ot_desc,c_estado"
    )
    df_ot = pd.DataFrame(ot_records)
    log.info(f"  orden_trabajo: {len(df_ot)} filas")

    log.info("Extrayendo ot_repuesto...")
    otr_records = fetch_all(
        client, "ot_repuesto",
        select="n_ot,producto_id,descripcion,cantidad"
    )
    df_otr = pd.DataFrame(otr_records)
    log.info(f"  ot_repuesto: {len(df_otr)} filas")

    log.info("Extrayendo repuesto...")
    rep_records = fetch_all(
        client, "repuesto",
        select="c_repuesto,descripcion,marca"
    )
    df_rep = pd.DataFrame(rep_records)
    log.info(f"  repuesto: {len(df_rep)} filas")

    return df_ot, df_otr, df_rep


def transform(df_ot: pd.DataFrame, df_otr: pd.DataFrame, df_rep: pd.DataFrame) -> pd.DataFrame:
    """
    Join y agregación mensual.
    Resultado: una fila = un repuesto × un mes × un año → cantidad total demandada.
    """

    # ── 1. Limpiar claves de join ────────────────────────────────────────────
    df_ot["n_ot"] = df_ot["n_ot"].astype(str).str.strip()
    df_otr["n_ot"] = df_otr["n_ot"].astype(str).str.strip()
    df_otr["producto_id"] = df_otr["producto_id"].astype(str).str.strip()

    # ── 2. Descartar OTs sin datos de repuesto ───────────────────────────────
    before = len(df_otr)
    df_otr = df_otr[df_otr["n_ot"].notna() & (df_otr["n_ot"] != "") & (df_otr["n_ot"] != "0")]
    df_otr = df_otr[df_otr["producto_id"].notna() & (df_otr["producto_id"] != "")]
    log.info(f"  Filas válidas en ot_repuesto: {before} → {len(df_otr)}")

    # ── 3. Join ot_repuesto ⋈ orden_trabajo ─────────────────────────────────
    df_merged = pd.merge(
        df_otr[["n_ot", "producto_id", "descripcion", "cantidad"]],
        df_ot[["n_ot", "anio", "mes", "km", "tipo_ot_desc"]],
        on="n_ot",
        how="inner"
    )
    log.info(f"  Filas después del join OT: {len(df_merged)}")

    # ── 4. Join con repuesto para descripción oficial ────────────────────────
    df_merged = pd.merge(
        df_merged,
        df_rep[["c_repuesto", "descripcion", "marca"]].rename(
            columns={"descripcion": "descripcion_repuesto", "marca": "marca_repuesto"}
        ),
        left_on="producto_id",
        right_on="c_repuesto",
        how="left"
    )

    # ── 5. Coerce numéricos ──────────────────────────────────────────────────
    df_merged["cantidad"] = pd.to_numeric(df_merged["cantidad"], errors="coerce").fillna(0)
    df_merged["km"] = pd.to_numeric(df_merged["km"], errors="coerce").fillna(0)
    df_merged["mes"] = pd.to_numeric(df_merged["mes"], errors="coerce")
    df_merged["anio"] = pd.to_numeric(df_merged["anio"], errors="coerce")

    # Descartar filas sin mes/año válidos
    df_merged = df_merged.dropna(subset=["mes", "anio"])
    df_merged["mes"] = df_merged["mes"].astype(int)
    df_merged["anio"] = df_merged["anio"].astype(int)

    # ── 6. Guardar raw antes de agregar (útil para debug) ───────────────────
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(RAW_CSV, index=False, encoding="utf-8")
    log.info(f"  Raw guardado en {RAW_CSV} ({len(df_merged)} filas)")

    # ── 7. Agregación mensual ────────────────────────────────────────────────
    #  Agrupamos por (repuesto × mes × año) y:
    #    - sumamos cantidad (demanda total del período)
    #    - promediamos km (contexto del vehículo)
    #    - tomamos el tipo_ot más frecuente (moda)
    agg = df_merged.groupby(["producto_id", "mes", "anio"]).agg(
        descripcion_repuesto=("descripcion_repuesto", "first"),
        marca_repuesto=("marca_repuesto", "first"),
        cantidad_total=("cantidad", "sum"),
        km_promedio=("km", "mean"),
        tipo_ot_desc=("tipo_ot_desc", lambda x: x.mode()[0] if len(x) > 0 else ""),
        n_ots=("n_ot", "count"),   # número de OTs que generaron esta demanda
    ).reset_index()

    log.info(f"  Filas agregadas (repuesto × mes × año): {len(agg)}")
    log.info(f"  Repuestos únicos: {agg['producto_id'].nunique()}")
    log.info(f"  Rango de fechas: {agg['anio'].min()}/{agg['mes'].min()} "
             f"→ {agg['anio'].max()}/{agg['mes'].max()}")
    log.info(f"  Demanda promedio por fila: {agg['cantidad_total'].mean():.2f} "
             f"(max: {agg['cantidad_total'].max():.0f})")

    return agg


def save(df: pd.DataFrame) -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"✅ CSV guardado en {OUTPUT_CSV} ({len(df)} filas)")


def main():
    log.info("=== ETL AutoX Insight: Supabase → demanda_mensual.csv ===")
    client = get_supabase_client()
    df_ot, df_otr, df_rep = extract(client)
    df_demanda = transform(df_ot, df_otr, df_rep)
    save(df_demanda)

    # Resumen final para que el usuario vea que data tiene
    print("\n=== RESUMEN DEL DATASET ===")
    print(f"   Filas totales (repuesto x mes x anio): {len(df_demanda)}")
    print(f"   Repuestos unicos: {df_demanda['producto_id'].nunique()}")
    print(f"   Tipos de OT unicos: {df_demanda['tipo_ot_desc'].nunique()}")
    print(f"   Cantidad minima: {df_demanda['cantidad_total'].min():.0f}")
    print(f"   Cantidad maxima: {df_demanda['cantidad_total'].max():.0f}")
    print(f"   Cantidad media: {df_demanda['cantidad_total'].mean():.2f}")
    print(f"\n   Top 5 repuestos mas demandados:")
    top5 = (
        df_demanda.groupby(["producto_id", "descripcion_repuesto"])["cantidad_total"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    for (pid, desc), qty in top5.items():
        print(f"   * {pid} | {desc} -> {qty:.0f} unidades totales")
    print()


if __name__ == "__main__":
    main()
