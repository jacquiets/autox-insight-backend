from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from supabase import Client

from app.core.config import settings
from app.core.supabase import get_supabase
from app.core.supabase_admin import get_supabase_admin
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    PasswordResetConfirm,
    PasswordResetRequest,
    UserPublic,
)
from app.services.auth import login_user

router = APIRouter()

# Nombre de la cookie que almacenara el JWT.
# HttpOnly = JavaScript no puede leerla - protege contra XSS.
# Secure   = Solo se envia por HTTPS en produccion.
# SameSite = "none" permite enviar la cookie en peticiones cross-site.
AUTH_COOKIE_NAME = "autox_access_token"


@router.post("/login", response_model=LoginResponse, summary="Iniciar sesion")
async def login(
    credentials: LoginRequest,
    response: Response,
    supabase: Client = Depends(get_supabase),
) -> LoginResponse:
    """
    Autentica al usuario y devuelve sus datos de perfil.

    - El token JWT se almacena en una cookie HttpOnly (no en el cuerpo
      de la respuesta), de modo que el frontend nunca puede accederlo
      desde JavaScript y no necesita guardarlo en localStorage.
    - El frontend debe enviar las cookies en cada peticion subsecuente
      usando credentials: include en fetch.
    """
    access_token, user_public = await login_user(credentials, supabase)

    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=60 * 60 * 8,
        path="/",
    )

    return LoginResponse(
        message="Sesion iniciada correctamente.",
        access_token=access_token,
        user=user_public,
    )


@router.post("/logout", summary="Cerrar sesion")
async def logout(
    response: Response,
    supabase: Client = Depends(get_supabase),
):
    """
    Cierra la sesion eliminando la cookie HttpOnly del navegador.
    """
    supabase.auth.sign_out()
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
    return {"message": "Sesion cerrada correctamente."}


@router.get("/me", response_model=UserPublic, summary="Perfil del usuario autenticado")
async def get_me(
    supabase: Client = Depends(get_supabase),
    autox_access_token: str | None = Cookie(default=None),
) -> UserPublic:
    """
    Verifica la cookie HttpOnly y devuelve el perfil del usuario.
    Utilizado por el frontend para rehidratar el estado de sesion
    tras un refresh de pagina, sin necesidad de localStorage.
    """
    if not autox_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No autenticado.",
        )

    try:
        user_response = supabase.auth.get_user(autox_access_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesion invalida o expirada.",
        )

    if not user_response or not user_response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesion invalida o expirada.",
        )

    user_id = user_response.user.id

    profile_response = (
        supabase.table("usuario")
        .select("correo_corporativo, nombre_completo, cargo")
        .eq("id", user_id)
        .single()
        .execute()
    )

    if not profile_response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Perfil de usuario no encontrado.",
        )

    data = profile_response.data
    return UserPublic(
        correo_corporativo=data["correo_corporativo"],
        nombre_completo=data["nombre_completo"],
        cargo=data["cargo"],
    )


@router.post(
    "/request-reset",
    summary="Solicitar email de recuperacion de contrasena",
    status_code=status.HTTP_200_OK,
)
async def request_password_reset(
    body: PasswordResetRequest,
    supabase: Client = Depends(get_supabase),
):
    """
    Envia un email de recuperacion de contrasena al usuario.

    Seguridad: siempre responde con 200 OK independientemente de si
    el email existe o no, para evitar revelar que cuentas estan registradas
    (user enumeration attack prevention).

    El enlace en el email apuntara a FRONTEND_URL/reset-password?token_hash=XXX&type=recovery
    """
    redirect_url = f"{settings.FRONTEND_URL}/reset-password"

    try:
        supabase.auth.reset_password_for_email(
            body.email,
            options={"redirect_to": redirect_url},
        )
    except Exception:
        pass  # No revelar errores internos al cliente

    return {
        "message": "Si el correo esta registrado, recibiras un enlace para restablecer tu contrasena."
    }


@router.post(
    "/confirm-reset",
    summary="Confirmar el nuevo password usando el token del email",
    status_code=status.HTTP_200_OK,
)
async def confirm_password_reset(
    body: PasswordResetConfirm,
):
    """
    Actualiza la contrasena del usuario usando el access_token obtenido
    cuando Supabase JS SDK verifico el OTP del enlace del email en el frontend.

    Flujo:
    1. El frontend llama a supabase.auth.verifyOtp({ token_hash, type: recovery })
       que devuelve un access_token valido.
    2. El frontend envia ese token + la nueva contrasena a este endpoint.
    3. El backend usa el cliente admin para obtener el user_id del token
       y actualizar la contrasena.

    Requiere SUPABASE_SERVICE_KEY configurada en Railway.
    """
    supabase_admin = get_supabase_admin()

    # 1. Validar el token y obtener el user_id
    try:
        user_response = supabase_admin.auth.get_user(body.access_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El enlace de recuperacion es invalido o ha expirado.",
        )

    if not user_response or not user_response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El enlace de recuperacion es invalido o ha expirado.",
        )

    user_id = user_response.user.id

    # 2. Actualizar la contrasena usando el cliente admin (service_role)
    try:
        supabase_admin.auth.admin.update_user_by_id(
            user_id,
            {"password": body.new_password},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No se pudo actualizar la contrasena: {exc}",
        )

    return {"message": "Contrasena actualizada correctamente. Ya puedes iniciar sesion."}
