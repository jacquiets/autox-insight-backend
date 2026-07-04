"""
PRUEBAS UNITARIAS — Lógica pura del módulo de IA.

Prueban funciones individuales de forma aislada (sin HTTP ni base de datos):
cálculo de confianza (RF-10), métrica wMAPE, gate de calidad (RNF-02) y
construcción del vector de features.
"""
import numpy as np
import pytest

from app.api.v1.endpoints import ml
from app.api.v1.endpoints.ml import (
    _compute_confidence, _label_for, _explain,
    CONFIDENCE_HIGH_THRESHOLD,
)


# ── Confianza (RF-10) ─────────────────────────────────────────────────────────

class TestConfianza:
    """El cálculo de confianza debe reflejar la densidad histórica del SKU."""

    def test_repuesto_nuevo_es_extrapolacion(self, model_bundle):
        """Un SKU jamás visto → confianza baja (extrapolación)."""
        conf, obs = _compute_confidence("SKU-INEXISTENTE-XYZ", cantidad=10.0)
        assert obs == 0
        assert conf <= 0.5, "Un repuesto nuevo no puede tener confianza alta"

    def test_repuesto_conocido_con_historia_es_alta_confianza(self, model_bundle):
        """Un SKU con mucha historia debe superar el umbral del 80%."""
        # Tomamos un SKU real con muchas observaciones del bundle.
        perfil = model_bundle["encoder"]["sku_profile"]
        sku_solido = max(perfil.items(), key=lambda kv: kv[1]["obs_count"])[0]
        conf, obs = _compute_confidence(sku_solido, cantidad=20.0)
        assert obs >= 4
        assert conf >= CONFIDENCE_HIGH_THRESHOLD, (
            f"SKU con {obs} meses de historia debería ser Alta Confiabilidad"
        )

    def test_confianza_acotada_entre_0_y_1(self, model_bundle):
        """La confianza nunca debe salirse de [0, 1]."""
        for sku in ["SKU-FALSO", *list(model_bundle["encoder"]["sku_profile"])[:20]]:
            conf, _ = _compute_confidence(sku, cantidad=5.0)
            assert 0.0 <= conf <= 1.0

    def test_mas_historia_implica_mas_o_igual_confianza(self, model_bundle):
        """Monotonía: entre dos SKUs, el de más historia no debe tener menos confianza."""
        perfil = model_bundle["encoder"]["sku_profile"]
        ordenados = sorted(perfil.items(), key=lambda kv: kv[1]["obs_count"])
        bajo = ordenados[0][0]
        alto = ordenados[-1][0]
        c_bajo, _ = _compute_confidence(bajo, 10.0)
        c_alto, _ = _compute_confidence(alto, 10.0)
        assert c_alto >= c_bajo


# ── Etiquetas de confiabilidad (RF-10) ────────────────────────────────────────

class TestEtiquetas:
    def test_umbral_alta_confiabilidad(self):
        assert _label_for(0.95) == "Alta Confiabilidad"
        assert _label_for(CONFIDENCE_HIGH_THRESHOLD) == "Alta Confiabilidad"

    def test_confianza_media(self):
        assert _label_for(0.70) == "Confianza Media"

    def test_extrapolacion_baja(self):
        assert _label_for(0.40) == "Extrapolación (baja confianza)"

    def test_explicacion_repuesto_nuevo_advierte(self):
        texto = _explain(0.40, obs=0, conocido=False)
        assert "NUEVO" in texto or "extrapolación" in texto.lower()


# ── Métrica wMAPE (RNF-02) ────────────────────────────────────────────────────

class TestWMAPE:
    """El wMAPE del pipeline de entrenamiento."""

    def _wmape(self, y_true, y_pred):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_mod",
            str(__import__("pathlib").Path(ml.__file__).resolve().parents[4] / "ml" / "train.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._wmape(np.array(y_true), np.array(y_pred))

    def test_prediccion_perfecta_da_cero(self):
        assert self._wmape([10, 20, 30], [10, 20, 30]) == 0.0

    def test_error_conocido(self):
        # |1|+|1|+|1| = 3 sobre suma real 60 → 5%
        assert self._wmape([10, 20, 30], [11, 21, 31]) == pytest.approx(5.0, abs=0.01)

    def test_suma_cero_no_rompe(self):
        assert self._wmape([0, 0], [0, 0]) == 0.0


# ── Gate de calidad (RNF-02) ──────────────────────────────────────────────────

class TestGate:
    def _load_train(self):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "train_mod2",
            str(pathlib.Path(ml.__file__).resolve().parents[4] / "ml" / "train.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_gate_promueve_si_cumple(self):
        train = self._load_train()
        assert train.passes_gate({"wmape": 30.0, "wmape_gate": 60.0}) is True

    def test_gate_bloquea_si_degrada(self):
        train = self._load_train()
        assert train.passes_gate({"wmape": 90.0, "wmape_gate": 60.0}) is False


# ── Construcción del vector de features ───────────────────────────────────────

class TestFeatures:
    def test_vector_respeta_orden_del_bundle(self, model_bundle):
        """El vector de inferencia debe tener tantas columnas como feature_cols."""
        from app.api.v1.endpoints.ml import PredictRequest, _predict_one
        req = PredictRequest(codigo_repuesto="X", mes=6, anio=2026, km=50000)
        # No debe lanzar excepción de shape aunque el SKU no exista.
        resp = _predict_one(req)
        assert resp.cantidad_estimada >= 0

    def test_cantidad_nunca_negativa(self, model_bundle):
        from app.api.v1.endpoints.ml import PredictRequest, _predict_one
        for mes in range(1, 13):
            resp = _predict_one(PredictRequest(codigo_repuesto="X", mes=mes, km=0))
            assert resp.cantidad_estimada >= 0
