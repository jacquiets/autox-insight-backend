# Resultados de Pruebas — Backend (AutoX Insight)

**Proyecto:** AutoX Insight — bpA Motors · SCM Intelligence
**Componente:** Backend FastAPI + XGBoost (`demand-forecast v3`)
**Fecha de ejecución:** 2026-07-03
**Entorno:** Python 3.12.7 · pytest 9.1.1 · Windows 10 · modelo `ml/model.pkl` v3.0 cargado en RAM
**Comando:** `python -m pytest tests/ -c tests/pytest.ini`

---

## Resumen ejecutivo

| Categoría | Pruebas | Pasan | Fallan | Estado |
|---|---:|---:|---:|:---:|
| Unitarias | 14 | 14 | 0 | ✅ |
| Funcionales | 15 | 15 | 0 | ✅ |
| Rendimiento | 4 | 4 | 0 | ✅ |
| **TOTAL** | **33** | **33** | **0** | ✅ **100%** |

> Las pruebas de **usabilidad** de la API (heurísticas de diseño de API) se documentan en la sección 4, ya que no son automatizables como asserts pero sí verificables.

---

## 1. Pruebas Unitarias (14) — lógica pura del módulo IA

Prueban funciones individuales de forma aislada, sin HTTP ni base de datos.

### 1.1 Cálculo de confianza (RF-10)
| Prueba | Qué valida | Resultado |
|---|---|:---:|
| `test_repuesto_nuevo_es_extrapolacion` | Un SKU nunca visto → confianza ≤ 0.5 y 0 observaciones | ✅ |
| `test_repuesto_conocido_con_historia_es_alta_confianza` | SKU con más historia → confianza ≥ 80% (umbral de negocio) | ✅ |
| `test_confianza_acotada_entre_0_y_1` | La confianza nunca se sale de [0, 1] en 20+ SKUs | ✅ |
| `test_mas_historia_implica_mas_o_igual_confianza` | Monotonía: más historia ⇒ confianza ≥ | ✅ |

### 1.2 Etiquetas de confiabilidad (RF-10)
| Prueba | Qué valida | Resultado |
|---|---|:---:|
| `test_umbral_alta_confiabilidad` | ≥ 0.80 ⇒ "Alta Confiabilidad" | ✅ |
| `test_confianza_media` | 0.70 ⇒ "Confianza Media" | ✅ |
| `test_extrapolacion_baja` | 0.40 ⇒ "Extrapolación (baja confianza)" | ✅ |
| `test_explicacion_repuesto_nuevo_advierte` | La explicación de un SKU nuevo advierte al usuario | ✅ |

### 1.3 Métrica wMAPE (RNF-02)
| Prueba | Qué valida | Resultado |
|---|---|:---:|
| `test_prediccion_perfecta_da_cero` | Predicción exacta ⇒ wMAPE = 0% | ✅ |
| `test_error_conocido` | Error de 3/60 ⇒ wMAPE = 5.0% (exacto) | ✅ |
| `test_suma_cero_no_rompe` | División por cero controlada ⇒ 0.0 | ✅ |

### 1.4 Gate de calidad (RNF-02) y features
| Prueba | Qué valida | Resultado |
|---|---|:---:|
| `test_gate_promueve_si_cumple` | wMAPE 30% ≤ gate 60% ⇒ promueve | ✅ |
| `test_gate_bloquea_si_degrada` | wMAPE 90% > gate 60% ⇒ bloquea | ✅ |
| `test_vector_respeta_orden_del_bundle` | El vector de inferencia respeta `feature_cols` | ✅ |
| `test_cantidad_nunca_negativa` | La predicción nunca es negativa (los 12 meses) | ✅ |

---

## 2. Pruebas Funcionales (15) — endpoints HTTP

Prueban el comportamiento observable vía `TestClient`. Supabase mockeado; modelo ML real.

### 2.1 Salud general
| Prueba | Endpoint | Resultado |
|---|---|:---:|
| `test_health` | `GET /api/v1/health` → 200 | ✅ |
| `test_root` | `GET /` → 200 | ✅ |

### 2.2 Estado del modelo IA (RF-11)
| Prueba | Qué valida | Resultado |
|---|---|:---:|
| `test_status_modelo_cargado` | `GET /ml/status` reporta modelo cargado, algoritmo, repuestos | ✅ |
| `test_status_expone_metricas` | Expone wMAPE, MAPE alta rotación y MAE | ✅ |
| `test_status_umbral_confiabilidad` | Umbral de alta confiabilidad = 0.80 | ✅ |

### 2.3 Predicción con confianza real (RF-09 / RF-10)
| Prueba | Qué valida | Resultado |
|---|---|:---:|
| `test_predice_repuesto_conocido` | SKU conocido → predicción válida + confianza [0,1] | ✅ |
| `test_respuesta_incluye_etiqueta_y_explicacion` | La respuesta trae etiqueta + explicación + observaciones | ✅ |
| `test_repuesto_nuevo_marca_extrapolacion` | SKU inexistente → `alta_confiabilidad=false` | ✅ |
| `test_km_influye_en_el_contexto` | El kilometraje se acepta como variable de contexto (RF-09) | ✅ |
| `test_mes_fuera_de_rango_es_rechazado` | mes=13 → 422 (validación de contrato) | ✅ |
| `test_km_negativo_es_rechazado` | km=-5 → 422 | ✅ |

### 2.4 Seguridad de endpoints protegidos (RF-12, RF-15)
| Prueba | Qué valida | Resultado |
|---|---|:---:|
| `test_retrain_requiere_autenticacion` | `POST /ml/retrain` sin token → 401 | ✅ |
| `test_generar_oc_requiere_autenticacion` | `POST /purchase-orders/generate` sin token → 401 | ✅ |
| `test_suggestions_requiere_autenticacion` | `GET /purchase-orders/suggestions` sin token → 401 | ✅ |

---

## 3. Pruebas de Rendimiento (4) — latencia de inferencia (RNF-03)

**Requerimiento (RNF-03 / Objetivo 4):** la inferencia debe completarse en **< 1.5 s**.

### Métricas medidas (200 inferencias, tras warm-up)

| Métrica | Valor | Umbral RNF-03 | Estado |
|---|---:|---:|:---:|
| Latencia mínima | 10.73 ms | — | — |
| Latencia media | 29.06 ms | — | — |
| **Percentil 50 (mediana)** | **14.81 ms** | 1500 ms | ✅ **101× más rápido** |
| **Percentil 95** | **121.88 ms** | 1500 ms | ✅ **12× más rápido** |
| Percentil 99 | 273.06 ms | 1500 ms | ✅ |
| Latencia máxima | 315.91 ms | 1500 ms | ✅ |
| **Throughput (1 worker)** | **34.4 req/s** | — | ✅ |

> **Interpretación:** el modelo montado en RAM (bundle pickle) responde en **~15 ms de mediana**, dos órdenes de magnitud por debajo del umbral de negocio. Incluso el peor caso medido (316 ms) queda holgadamente dentro del 1.5 s. Nota: estas mediciones son locales (sin latencia de red); en producción (Railway) hay que sumar el RTT de red, pero el margen es amplísimo.

| Prueba | Qué valida | Resultado |
|---|---|:---:|
| `test_prediccion_individual_bajo_umbral` | 1 inferencia < 1.5 s | ✅ |
| `test_percentil_95_de_100_inferencias` | p95 de 100 inferencias < 1.5 s | ✅ |
| `test_status_es_rapido` | `/ml/status` < 0.3 s | ✅ |
| `test_throughput_secuencial` | > 20 req/s en 1 worker | ✅ |

---

## 4. Usabilidad de la API (evaluación heurística)

La "usabilidad" de un backend se mide por qué tan fácil es consumirlo correctamente. Evaluado contra buenas prácticas de diseño de API REST:

| Heurística | Cumple | Evidencia |
|---|:---:|---|
| **Mensajes de error claros** | ✅ | 401 con `detail` explicando "No estás autenticado"; 503 con instrucción de verificar `model.pkl`; 422 de Pydantic con el campo inválido |
| **Validación de contrato en el borde** | ✅ | `mes` (1-12) y `km` (≥0) validados por Pydantic → 422 automático antes de tocar el modelo |
| **Respuestas autoexplicativas** | ✅ | La predicción incluye `etiqueta_confianza` y `explicacion` en lenguaje humano, no solo números crudos |
| **Documentación viva** | ✅ | OpenAPI/Swagger en `/api/v1/docs` con summaries por endpoint y descripción de cada campo |
| **Nombres consistentes** | ✅ | Convención `snake_case`, prefijos por dominio (`/ml`, `/purchase-orders`), tags agrupados |
| **Degradación controlada** | ✅ | Si el modelo no está cargado, `/predict` responde 503 (no 500) y `/status` reporta `modelo_cargado:false` |

---

## Cómo reproducir

```bash
cd autox-insight-backend
pip install pytest httpx          # dependencias de test
python -m pytest tests/ -c tests/pytest.ini -v
```

## Cobertura de requerimientos

| Requerimiento | Cubierto por |
|---|---|
| RF-09 (predicción con km) | Funcional 2.3 |
| RF-10 (confianza real + etiqueta) | Unitaria 1.1/1.2 + Funcional 2.3 |
| RF-11 (estado del modelo) | Funcional 2.2 |
| RF-12 (OC inteligente — seguridad) | Funcional 2.4 |
| RF-15 (reentrenamiento — seguridad + gate) | Funcional 2.4 + Unitaria 1.4 |
| RNF-02 (gate de calidad wMAPE) | Unitaria 1.3/1.4 |
| RNF-03 (inferencia < 1.5 s) | Rendimiento 3 |
