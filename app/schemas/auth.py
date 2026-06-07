from pydantic import BaseModel, EmailStr


# ── Request ──────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ── Response ─────────────────────────────────────────────────────────────────
class UserPublic(BaseModel):
    """Datos del usuario que se exponen al frontend para mostrar en pantalla."""
    correo_corporativo: str
    nombre_completo: str
    cargo: str


class LoginResponse(BaseModel):
    message: str
    access_token: str
    user: UserPublic
