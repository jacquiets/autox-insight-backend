# ── Cliente Supabase Admin ────────────────────────────────────────────────────
# Usa la service_role key para operaciones privilegiadas (actualizar contraseñas).
# Este cliente bypasea Row Level Security — usarlo SOLO en el backend,
# NUNCA exponer la key en el frontend.

from supabase import create_client, Client
from app.core.config import settings


def get_supabase_admin() -> Client:
    """
    Retorna un cliente Supabase con permisos de administrador.
    Necesario para admin.update_user_by_id() al resetear contrasenas.

    Importante: instanciar bajo demanda (no globalmente) para evitar
    errores de startup si SUPABASE_SERVICE_KEY no esta configurada.
    """
    if not settings.SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "SUPABASE_SERVICE_KEY no esta configurada. "
            "Agregala como variable de entorno en Railway."
        )

    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
