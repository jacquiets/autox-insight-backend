"""
PRUEBAS FUNCIONALES — Endpoints HTTP de la API.

Prueban el comportamiento observable de los endpoints vía TestClient:
contratos de respuesta, códigos de estado, y los requerimientos RF-09,
RF-10, RF-11. Supabase está mockeado; el modelo ML es real.
"""
import pytest


# ── Salud general ─────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "status" in r.json()


# ── RF-11: Estado del modelo IA ───────────────────────────────────────────────

class TestEstadoModelo:
    def test_status_modelo_cargado(self, client):
        r = client.get("/api/v1/ml/status")
        assert r.status_code == 200
        body = r.json()
        assert body["modelo_cargado"] is True
        assert body["algoritmo"] == "XGBoost Regressor"
        assert body["repuestos_conocidos"] > 0

    def test_status_expone_metricas(self, client):
        """RF-11: el status debe exponer las métricas de salud del modelo."""
        r = client.get("/api/v1/ml/status")
        metrics = r.json()["metrics"]
        assert metrics is not None
        assert "wmape" in metrics
        assert "mape_alta_rotacion" in metrics
        assert "mae" in metrics

    def test_status_umbral_confiabilidad(self, client):
        r = client.get("/api/v1/ml/status")
        assert r.json()["umbral_alta_confiabilidad"] == 0.80


# ── RF-09 / RF-10: Predicción con confianza real ──────────────────────────────

class TestPrediccion:
    def _un_sku_conocido(self, client):
        return client.get("/api/v1/ml/status").json()  # solo para asegurar carga

    def test_predice_repuesto_conocido(self, client, model_bundle):
        sku = next(iter(model_bundle["encoder"]["repuesto_map"]))
        r = client.post("/api/v1/ml/predict", json={
            "codigo_repuesto": sku, "mes": 6, "anio": 2026, "km": 50000,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["repuesto_conocido"] is True
        assert body["cantidad_estimada"] >= 0
        assert 0.0 <= body["confianza"] <= 1.0

    def test_respuesta_incluye_etiqueta_y_explicacion(self, client, model_bundle):
        """RF-10: la respuesta debe traer etiqueta de confiabilidad y explicación."""
        sku = next(iter(model_bundle["encoder"]["repuesto_map"]))
        r = client.post("/api/v1/ml/predict", json={"codigo_repuesto": sku, "mes": 6})
        body = r.json()
        assert body["etiqueta_confianza"] in (
            "Alta Confiabilidad", "Confianza Media", "Extrapolación (baja confianza)",
        )
        assert isinstance(body["explicacion"], str) and len(body["explicacion"]) > 10
        assert "alta_confiabilidad" in body
        assert "observaciones_historicas" in body

    def test_repuesto_nuevo_marca_extrapolacion(self, client):
        """RF-10: un SKU inexistente debe advertirse como extrapolación."""
        r = client.post("/api/v1/ml/predict", json={
            "codigo_repuesto": "SKU-QUE-NO-EXISTE-123", "mes": 6, "km": 50000,
        })
        body = r.json()
        assert body["repuesto_conocido"] is False
        assert body["alta_confiabilidad"] is False

    def test_km_influye_en_el_contexto(self, client, model_bundle):
        """RF-09: el kilometraje es una variable de contexto aceptada."""
        sku = next(iter(model_bundle["encoder"]["repuesto_map"]))
        r0 = client.post("/api/v1/ml/predict", json={"codigo_repuesto": sku, "mes": 6, "km": 0})
        r1 = client.post("/api/v1/ml/predict", json={"codigo_repuesto": sku, "mes": 6, "km": 200000})
        assert r0.status_code == 200 and r1.status_code == 200

    def test_mes_fuera_de_rango_es_rechazado(self, client):
        """Validación de contrato: mes debe estar en 1-12."""
        r = client.post("/api/v1/ml/predict", json={"codigo_repuesto": "X", "mes": 13})
        assert r.status_code == 422  # Unprocessable Entity

    def test_km_negativo_es_rechazado(self, client):
        r = client.post("/api/v1/ml/predict", json={"codigo_repuesto": "X", "mes": 6, "km": -5})
        assert r.status_code == 422


# ── Seguridad de endpoints protegidos (RF-12, RF-15) ──────────────────────────

class TestSeguridad:
    def test_retrain_requiere_autenticacion(self, client):
        """RF-15: reentrenar sin token debe dar 401."""
        r = client.post("/api/v1/ml/retrain", json={})
        assert r.status_code == 401

    def test_generar_oc_requiere_autenticacion(self, client):
        """RF-12: generar OC sin token debe dar 401."""
        r = client.post("/api/v1/purchase-orders/generate", json={
            "items": [{"codigo_repuesto": "X", "compra_sugerida": 5}],
        })
        assert r.status_code == 401

    def test_suggestions_requiere_autenticacion(self, client):
        r = client.get("/api/v1/purchase-orders/suggestions?mes=6")
        assert r.status_code == 401
