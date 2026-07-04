"""
PRUEBAS DE RENDIMIENTO — Latencia de inferencia del modelo IA.

Verifica el requerimiento RNF-03 / Objetivo 4: la inferencia debe completarse
en menos de 1.5 segundos. Mide latencia individual, p95 y throughput.
Los umbrales son holgados respecto al objetivo porque en local no hay red.
"""
import time
import statistics

import pytest

# Umbral de negocio (RNF-03 / Objetivo 4): inferencia < 1.5 s
UMBRAL_INFERENCIA_S = 1.5
# En local (sin red) exigimos mucho más rápido para detectar regresiones.
UMBRAL_LOCAL_S = 0.5


def _sku(model_bundle):
    return next(iter(model_bundle["encoder"]["repuesto_map"]))


class TestLatenciaInferencia:
    def test_prediccion_individual_bajo_umbral(self, client, model_bundle):
        """Una predicción debe responder muy por debajo de 1.5 s."""
        sku = _sku(model_bundle)
        # Warm-up (excluye el costo del primer JIT/allocación)
        client.post("/api/v1/ml/predict", json={"codigo_repuesto": sku, "mes": 6})

        t0 = time.perf_counter()
        r = client.post("/api/v1/ml/predict", json={"codigo_repuesto": sku, "mes": 6, "km": 50000})
        elapsed = time.perf_counter() - t0

        assert r.status_code == 200
        assert elapsed < UMBRAL_INFERENCIA_S, f"Inferencia tardó {elapsed:.3f}s (>1.5s RNF-03)"
        # Guardamos la métrica para el reporte
        TestLatenciaInferencia.latencia_individual = elapsed

    def test_percentil_95_de_100_inferencias(self, client, model_bundle):
        """El p95 de 100 inferencias debe mantenerse rápido (sin degradación)."""
        sku = _sku(model_bundle)
        tiempos = []
        for i in range(100):
            t0 = time.perf_counter()
            client.post("/api/v1/ml/predict", json={
                "codigo_repuesto": sku, "mes": (i % 12) + 1, "km": 40000 + i * 100,
            })
            tiempos.append(time.perf_counter() - t0)

        p50 = statistics.median(tiempos)
        p95 = statistics.quantiles(tiempos, n=20)[18]  # percentil 95
        p_max = max(tiempos)

        assert p95 < UMBRAL_INFERENCIA_S, f"p95={p95:.3f}s supera 1.5s"
        # Métricas para el reporte
        TestLatenciaInferencia.p50 = p50
        TestLatenciaInferencia.p95 = p95
        TestLatenciaInferencia.p_max = p_max
        TestLatenciaInferencia.n = len(tiempos)

    def test_status_es_rapido(self, client):
        """El health-check del modelo (RF-11) debe ser casi instantáneo."""
        t0 = time.perf_counter()
        r = client.get("/api/v1/ml/status")
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200
        assert elapsed < 0.3


def test_throughput_secuencial(client, model_bundle):
    """Throughput: cuántas inferencias por segundo soporta un solo worker."""
    sku = _sku(model_bundle)
    N = 200
    t0 = time.perf_counter()
    for i in range(N):
        client.post("/api/v1/ml/predict", json={"codigo_repuesto": sku, "mes": (i % 12) + 1})
    total = time.perf_counter() - t0
    rps = N / total
    assert rps > 20, f"Throughput bajo: {rps:.1f} req/s"
