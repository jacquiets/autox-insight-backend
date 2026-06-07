from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import Client

from app.core.supabase import get_supabase

# HTTPBearer extrae el header 'Authorization: Bearer <token>'
# Lo marcamos con auto_error=False para que no falle inmediatamente si no está,
# ya que también queremos buscar en la cookie HttpOnly.
bearer_scheme = HTTPBearer(auto_error=False)

def get_current_user_token(
    request: Request,
    auth: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    supabase: Client = Depends(get_supabase),
) -> str:
    """
    Dependencia de seguridad que obtiene el JWT validado, ya sea del 
    header Authorization (Bearer) o de la cookie HttpOnly.
    """
    token = None

    # 1. Intentar con Bearer Token (Swagger, Apps Móviles, Postman)
    if auth and auth.credentials:
        token = auth.credentials
    # 2. Fallback a Cookie HttpOnly (Frontend Web)
    elif "autox_access_token" in request.cookies:
        token = request.cookies.get("autox_access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No estás autenticado. Provee un token válido.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Validar el token contra Supabase Auth
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise ValueError("Token inválido o expirado")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
