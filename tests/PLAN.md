# Plan Integral de Ejecución de Pruebas — AutoX Insight (bpA Motors)

**Proyecto:** AutoX Insight — SCM Intelligence para bpA Motors
**Alcance:** Backend (FastAPI + XGBoost) y Frontend (React SPA)
**Este documento:** plan del **Backend**. El plan espejo del Frontend está en `autox-insight-X/tests/PLAN.md`.

---

## 1. Objetivo

Verificar que el módulo de IA y su API cumplen los requerimientos funcionales y no
funcionales del proyecto (RF-09 a RF-15, RNF-02, RNF-03) mediante cuatro tipos de
pruebas: **funcionales, unitarias, de rendimiento y de usabilidad**.

## 2. Estrategia por tipo de prueba

| Tipo | Qué prueba | Herramienta | Archivo |
|---|---|---|---|
| **Unitarias** | Lógica pura aislada: confianza, wMAPE, gate, features | pytest | `test_unit_ml.py` |
| **Funcionales** | Endpoints HTTP de punta a punta (contratos, códigos, seguridad) | pytest + TestClient | `test_functional_api.py` |
| **Rendimiento** | Latencia de inferencia p50/p95, throughput (RNF-03) | pytest + `time.perf_counter` | `test_performance.py` |
| **Usabilidad** | Heurísticas de diseño de API (errores claros, docs, degradación) | Evaluación documentada | `RESULTADOS.md §4` |

## 3. Entorno de pruebas

- Python 3.12, `pytest`, `httpx`.
- **Supabase se mockea** (`conftest.py`) → no requiere credenciales reales; las
  pruebas de datos se simulan.
- **El modelo ML es real**: `ml/model.pkl` v3 se carga en RAM para probar inferencia
  y latencia auténticas.

## 4. Matriz de cobertura (requerimiento → prueba)

| Requerimiento | Tipo de prueba | Caso |
|---|---|---|
| RF-09 (predicción con km) | Funcional | `test_km_influye_en_el_contexto` |
| RF-10 (confianza real + etiqueta) | Unitaria + Funcional | `TestConfianza`, `test_respuesta_incluye_etiqueta` |
| RF-11 (estado del modelo) | Funcional | `TestEstadoModelo` |
| RF-12 (OC inteligente) | Funcional (seguridad) | `test_generar_oc_requiere_autenticacion` |
| RF-15 (reentrenamiento) | Funcional + Unitaria | `test_retrain_requiere_autenticacion`, `TestGate` |
| RNF-02 (gate de calidad wMAPE) | Unitaria | `TestWMAPE`, `TestGate` |
| RNF-03 (inferencia < 1.5 s) | Rendimiento | `TestLatenciaInferencia` |

## 5. Criterios de aceptación

- ✅ 100% de las pruebas automatizadas pasan.
- ✅ p95 de inferencia < 1.5 s (RNF-03).
- ✅ El gate bloquea modelos que degradan la métrica (RNF-02).
- ✅ Los endpoints protegidos rechazan peticiones sin autenticación.

## 6. Ejecución

```bash
cd autox-insight-backend
pip install pytest httpx
python -m pytest tests/ -c tests/pytest.ini -v
```

## 7. Resultados

Ver **`RESULTADOS.md`** (mismo directorio) para el detalle completo con métricas.
**Última ejecución: 33/33 pruebas ✅ · p50 inferencia 14.8 ms · p95 121.9 ms.**

## 8. Riesgos y limitaciones conocidas

- Las mediciones de rendimiento son **locales** (sin latencia de red); en producción
  (Railway) hay que sumar el RTT, pero el margen respecto a 1.5 s es amplísimo.
- Las pruebas de datos usan Supabase mockeado; la integración real con la base de
  datos se valida manualmente en el entorno de staging.
- El objetivo de "MAPE < 10%" del documento **no es alcanzable** con demanda
  intermitente (ver `RESULTADOS.md`); por eso el gate usa wMAPE, métrica correcta
  para este régimen de datos.
