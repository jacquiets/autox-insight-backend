from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Supabase Backend"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    
    SUPABASE_URL: str = "https://your-project-id.supabase.co"
    SUPABASE_KEY: str = "your-supabase-anon-key"
    
    # Orígenes permitidos (frontend)
    ALLOWED_ORIGINS: list[str] | str = [
        "http://localhost:5173",  # Vite por defecto
        "http://localhost:3000",  # React/Next por defecto
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
