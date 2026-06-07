from pydantic import BaseModel, ConfigDict

class VehicleBrandSummary(BaseModel):
    marca: str
    modelo: str
    cantidad_vehiculos: int

    model_config = ConfigDict(from_attributes=True)
