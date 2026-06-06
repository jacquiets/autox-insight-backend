from supabase import create_client, Client
from app.core.config import settings

# Create a single global client instance
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# Dependency injection helper for FastAPI endpoints or services
def get_supabase() -> Client:
    return supabase
