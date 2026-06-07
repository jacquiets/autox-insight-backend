from fastapi import APIRouter, Depends, Response
from supabase import Client

from app.core.supabase import get_supabase
from app.schemas.auth import LoginRequest, LoginResponse
from app.services.auth import login_user

router = APIRouter()

# Nombre de la cookie que almacenará el JWT.
# HttpOnly = JavaScript no puede leerla → protege contra XSS.
# Secure   = Solo se envía por HTTPS en producción.
# SameSite = "lax" protege contra CSRF en la mayoría de casos.
AUTH_COOKIE_NAME = "autox_access_token"


@router.post("/login", response_model=LoginResponse, summary="Iniciar sesión")
async def login(
    credentials: LoginRequest,
    response: Response,
    supabase: Client = Depends(get_supabase),
) -> LoginResponse:
    """
    Autentica al usuario y devuelve sus datos de perfil.

    - El token JWT se almacena en una **cookie HttpOnly** (no en el cuerpo
      de la respuesta), de modo que el frontend nunca puede accederlo
      desde JavaScript y no necesita guardarlo en `localStorage`.
    - El frontend debe enviar las cookies en cada petición subsecuente
      usando `credentials: "include"` en `fetch` o `withCredentials: true`
      en axios.
    """
    access_token, user_public = await login_user(credentials, supabase)

    # Guardar el JWT en cookie HttpOnly — nunca viaja en el body de la respuesta
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        httponly=True,       # JavaScript no puede acceder a esta cookie
        secure=True,         # Solo HTTPS (en desarrollo local se puede poner False)
        samesite="lax",      # Protección básica contra CSRF
        max_age=60 * 60 * 8, # Expiración: 8 horas (en segundos)
        path="/",
    )

    return LoginResponse(
        message="Sesión iniciada correctamente.",
        user=user_public,
    )


@router.post("/logout", summary="Cerrar sesión")
async def logout(
    response: Response,
    supabase: Client = Depends(get_supabase),
):
    """
    Cierra la sesión eliminando la cookie HttpOnly del navegador.
    """
    supabase.auth.sign_out()
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
    return {"message": "Sesión cerrada correctamente."}
