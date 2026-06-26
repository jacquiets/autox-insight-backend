import os
import logging
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CLEAN = os.path.join(os.path.dirname(__file__), "..", "data", "clean")


# ── helpers ──────────────────────────────────────────────────────────────────

def read_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(RAW, filename)
    df = pd.read_csv(path, encoding="latin-1", dtype=str)
    log.info(f"Leído {filename}: {len(df)} filas")
    return df


def drop_log(before: int, after: int, reason: str) -> None:
    log.info(f"  Descartadas {before - after} filas ({reason}). Quedan {after}.")


def save_clean(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(CLEAN, name)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"Exportado {name}: {len(df)} filas → {path}")


# ── ETL principal ─────────────────────────────────────────────────────────────

def run_etl() -> dict[str, pd.DataFrame]:
    # 1. Lectura
    ot_cab = read_csv("export_OT_cabecera.csv")
    ot_prod = read_csv("export_OT_producto.csv")
    oc_det = read_csv("export_OC_detalle.csv")

    # 2. Strip en claves de join
    ot_prod["CODIGO"] = ot_prod["CODIGO"].str.strip()
    ot_prod["N_OT"] = ot_prod["N_OT"].str.strip()
    oc_det["C_REPUESTO"] = oc_det["C_REPUESTO"].str.strip()

    # 3. Descartar N_OT = 0 o vacío en OT_producto
    before = len(ot_prod)
    ot_prod = ot_prod[ot_prod["N_OT"].notna() & (ot_prod["N_OT"] != "") & (ot_prod["N_OT"] != "0")]
    drop_log(before, len(ot_prod), "N_OT = 0 o vacío en OT_producto")

    # Descartar filas sin CODIGO (sin código de repuesto no sirven para el modelo)
    before = len(ot_prod)
    ot_prod = ot_prod[ot_prod["CODIGO"].notna() & (ot_prod["CODIGO"].str.strip() != "")]
    drop_log(before, len(ot_prod), "CODIGO nulo o vacío en OT_producto")

    # 4. Normalizar tipos para join
    ot_cab["OT"] = ot_cab["OT"].str.strip()

    # 5. Join OT_cabecera ⟕ OT_producto  (OT = N_OT)
    log.info("Join OT_cabecera × OT_producto por OT = N_OT …")
    ot_merged = pd.merge(
        ot_cab,
        ot_prod,
        left_on="OT",
        right_on="N_OT",
        how="inner",
        suffixes=("_cab", "_prod"),
    )
    log.info(f"  Resultado join OT: {len(ot_merged)} filas")

    # 6. Join resultado ⟕ OC_detalle  (CODIGO = C_REPUESTO)
    log.info("Join resultado × OC_detalle por CODIGO = C_REPUESTO …")
    full = pd.merge(
        ot_merged,
        oc_det,
        left_on="CODIGO",
        right_on="C_REPUESTO",
        how="left",
        suffixes=("", "_oc"),
    )
    log.info(f"  Resultado join OC: {len(full)} filas")

    # 7. Guardar tablas limpias individuales + tabla completa
    save_clean(ot_cab, "ot_cabecera_clean.csv")
    save_clean(ot_prod, "ot_producto_clean.csv")
    save_clean(oc_det, "oc_detalle_clean.csv")
    save_clean(full, "ot_full_joined.csv")

    return {"ot_cabecera": ot_cab, "ot_producto": ot_prod, "oc_detalle": oc_det, "ot_full": full}


# ── Helpers de tipo ──────────────────────────────────────────────────────────

def _coerce_record(record: dict) -> dict:
    """Convierte strings a int/float/None para que PostgreSQL acepte los tipos."""
    out = {}
    for k, v in record.items():
        if v is None:
            out[k] = None
        elif isinstance(v, str):
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        else:
            out[k] = v
    return out


# ── Carga a Supabase ──────────────────────────────────────────────────────────

def load_to_supabase(tables: dict[str, pd.DataFrame]) -> None:
    load_dotenv()
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    supabase: Client = create_client(url, key)

    # mode: "upsert" usa PK natural | "insert" trunca primero por columna pk_col
    table_map = {
        "ot_cabecera": ("ot_cabecera",    "upsert", None),
        "ot_producto": ("ot_producto",    "insert", "id"),
        "oc_detalle":  ("oc_detalle",     "upsert", None),
        "ot_full":     ("ot_full_joined", "insert", "id"),
    }

    for key_name, (table_name, mode, pk_col) in table_map.items():
        df = tables[key_name].where(pd.notnull(tables[key_name]), None)
        df.columns = [c.lower() for c in df.columns]  # PostgreSQL guarda columnas en minúsculas
        records = [_coerce_record(r) for r in df.to_dict(orient="records")]

        # Truncar tablas con PK serial antes de reinsertar (idempotencia)
        if mode == "insert" and pk_col:
            log.info(f"Truncando '{table_name}' antes de insertar …")
            supabase.table(table_name).delete().gte(pk_col, 0).execute()

        log.info(f"Cargando {len(records)} registros en Supabase tabla '{table_name}' ({mode}) …")
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            if mode == "upsert":
                supabase.table(table_name).upsert(batch).execute()
            else:
                supabase.table(table_name).insert(batch).execute()
        log.info(f"  ✓ '{table_name}' cargada.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tables = run_etl()
    load_to_supabase(tables)
    log.info("ETL completo.")
