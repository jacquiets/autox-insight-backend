from datetime import date
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


# ── Sub-modelos ─────────────────────────────────────────────────────────────
class EstadoOT(BaseModel):
    c_estado: str
    descripcion: Optional[str] = None


class AutoInfo(BaseModel):
    vin: str
    placa: Optional[str] = None
    marca: Optional[str] = None
    modelo: Optional[str] = None
    anio_mod: Optional[int] = None
    tipo_transmision: Optional[str] = None
    tipo_combustible: Optional[str] = None


class WorkOrderItem(BaseModel):
    n_ot: str
    fecha: Optional[date] = None
    km: Optional[float] = None
    rango_km: Optional[str] = None
    tipo_ot_desc: Optional[str] = None
    requerimiento: Optional[str] = None
    dias_atencion: Optional[float] = None
    
    estado: Optional[EstadoOT] = None
    auto: Optional[AutoInfo] = None

    model_config = ConfigDict(from_attributes=True)


# ── Metadata y Response ─────────────────────────────────────────────────────
class WorkOrderMetadata(BaseModel):
    total_records: int
    current_page: int
    page_size: int
    total_pages: int


class WorkOrderListResponse(BaseModel):
    metadata: WorkOrderMetadata
    data: List[WorkOrderItem]


class WorkOrderPart(BaseModel):
    id: int
    producto_id: Optional[str] = None
    descripcion: Optional[str] = None
    marca: Optional[str] = None
    cantidad: Optional[float] = None
    precio_unitario: Optional[float] = None
    total: Optional[float] = None
    c_moneda: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
