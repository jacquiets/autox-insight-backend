"""
Configuración compartida de pytest para AutoX Insight backend.

Mockea el cliente de Supabase ANTES de importar la app, para que los tests
corran sin credenciales reales (las llamadas a datos se simulan por test).
El modelo ML (ml/model.pkl) sí se carga de verdad para probar inferencia real.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Mock de Supabase antes de que la app lo instancie ─────────────────────────
import supabase as _supabase_pkg  # noqa: E402

_fake_client = MagicMock(name="FakeSupabaseClient")
_supabase_pkg.create_client = lambda *a, **k: _fake_client


@pytest.fixture(scope="session")
def app():
    """Instancia la app FastAPI con el modelo ML cargado."""
    from app.main import app as fastapi_app
    from app.api.v1.endpoints.ml import load_model
    load_model()  # carga ml/model.pkl en RAM
    return fastapi_app


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture(scope="session")
def model_bundle():
    """El bundle del modelo ML ya cargado en memoria."""
    from app.api.v1.endpoints import ml
    ml.load_model()
    return ml._bundle
