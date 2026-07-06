from pydantic import BaseModel, EmailStr, field_validator


# ── Request ──────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class PasswordResetRequest(BaseModel):
    """Solicita el envío del email de recuperación de contraseña."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Confirma el nuevo password usando el access_token obtenido del enlace del email."""
    access_token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("La contraseña debe tener al menos 8 caracteres.")
        if not any(c.isdigit() for c in v):
            raise ValueError("La contraseña debe contener al menos un número.")
        if any(c in ";<>\"'" for c in v):
            raise ValueError("La contraseña contiene caracteres no permitidos.")
        return v


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
