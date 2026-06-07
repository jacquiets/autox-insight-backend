from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class PartItem(BaseModel):
    c_repuesto: str
    descripcion: Optional[str] = None
    marca: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class PartMetadata(BaseModel):
    total_records: int
    current_page: int
    page_size: int
    total_pages: int

class PartListResponse(BaseModel):
    metadata: PartMetadata
    data: List[PartItem]
