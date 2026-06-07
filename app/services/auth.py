from fastapi import HTTPException, status
from supabase import Client

from app.schemas.auth import LoginRequest, UserPublic


async def login_user(credentials: LoginRequest, supabase: Client) -> tuple[str, UserPublic]:
    """
    Autentica al usuario contra Supabase Auth y obtiene su perfil desde
    la tabla `usuario`.

    Retorna una tupla (access_token, UserPublic) para que el endpoint
    pueda almacenar el token en una cookie HttpOnly (nunca en el cuerpo
    de la respuesta, para evitar que el frontend lo guarde en localStorage).
    """

    # 1. Autenticar con Supabase Auth (email + password)
    try:
        auth_response = supabase.auth.sign_in_with_password(
            {"email": credentials.email, "password": credentials.password}
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas.",
        )

    if not auth_response.session or not auth_response.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas.",
        )

    access_token = auth_response.session.access_token

    # 2. Buscar el perfil del usuario en la tabla `usuario` usando su UUID de Supabase Auth
    user_id = auth_response.user.id

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

    user_data = profile_response.data
    user_public = UserPublic(
        correo_corporativo=user_data["correo_corporativo"],
        nombre_completo=user_data["nombre_completo"],
        cargo=user_data["cargo"],
    )

    return access_token, user_public
