# 💊 Análisis de Importaciones Farmacéuticas con Machine Learning
### Predicción de Atributos y Precio de Medicamentos Importados

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-9B59B6?style=for-the-badge&logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Estado](https://img.shields.io/badge/Estado-Demo_Funcional-27AE60?style=for-the-badge)](https://github.com/JuanDiego076/pharma-imports-ml)

---

## 🎯 Planteamiento del Problema

Las empresas que importan medicamentos enfrentan un reto operativo: decidir **qué importar, en qué cantidad y a qué costo esperado** sin visibilidad clara sobre los patrones históricos. Las decisiones basadas solo en intuición generan dos modos de fallo costosos: **desabastecimiento** (ventas perdidas + riesgo para el paciente) y **sobre-inventario** (costos de almacenamiento + vencimiento).

Este proyecto transforma ~27.000 registros históricos de importación en un **sistema de machine learning** que, dado el perfil logístico de una importación (vía de transporte, países de origen/procedencia/compra, pesos y valores), predice el **principio activo, fabricante, marca y precio** esperados — segmentado por tipo de producto farmacéutico.

**Conjunto de datos:** ~27.000 registros de importación · 9 variables de entrada · 4 objetivos (3 de clasificación + 1 de regresión)
**Segmentación:** 3 segmentos de producto (Farma, Oncológico, PT), con un modelo independiente por segmento.

---

## 📊 Resultados Principales

Todas las métricas se calculan sobre un **conjunto de prueba (20%)** separado del entrenamiento (`random_state=42`). Los precios se modelan con transformación logarítmica para manejar outliers, y las métricas de precio se reportan en la escala original (pesos).

### Clasificación — accuracy / F1 (ponderado)

| Objetivo | Farma | Oncológico | PT |
| :--- | :---: | :---: | :---: |
| Principio activo | 0.45 / 0.34 | **0.94 / 0.94** | 0.51 / 0.47 |
| Fabricante | 0.40 / 0.24 | **0.97 / 0.97** | 0.12 / 0.09 |
| Marca | 0.53 / 0.41 | **0.96 / 0.96** | 0.36 / 0.43 |

### Regresión de precio — mejora sobre baseline

| Segmento | R² | MAE | Mejora vs. baseline (media) |
| :--- | :---: | :---: | :---: |
| **Oncológico** | **0.73** | 195 | **+81%** |
| **PT** | **0.75** | 129 | **+78%** |
| Farma | 0.02 | 31.158 | +51% |

**Lectura de los resultados:** el segmento **Oncológico destaca con 94–97% de accuracy** y un R² de 0.73 en precio — con F1 ≈ accuracy, lo que indica acierto parejo entre clases, no sesgo hacia la mayoritaria. **PT** logra el mejor R² de precio (0.75). **Farma** es el caso más difícil: su alta dispersión de productos y precios limita el desempeño, especialmente en regresión (R² 0.02, apenas por encima del baseline). Reportar este contraste honestamente es parte del análisis: no todos los segmentos se comportan igual, y entender *por qué* es tan valioso como el número.

---

## 🏛️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE DATOS                            │
│                                                                 │
│  Datos crudos             Preprocesamiento        Variables     │
│  (~27.000 registros) ──►  - IDs/códigos como  ──►  9 features    │
│  39 columnas              categóricos (no num.)    Segmentado    │
│                           - log1p(precio)                       │
│                           - filtro precio < 0                   │
│                           - clases raras → OTROS               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   CAPA DE ENTRENAMIENTO  │
                    │  Segmento: TIPO_PRODUCTO │
                    │  ┌──────────────────┐   │
                    │  │ 3x LightGBM CLF  │   │  ← principio activo
                    │  │  (por segmento)  │   │  ← fabricante
                    │  │                  │   │  ← marca
                    │  ├──────────────────┤   │
                    │  │ 1x LightGBM REG  │   │  ← precio (log1p)
                    │  └──────────────────┘   │
                    │  Guardado: modelo_*.pkl  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   DASHBOARD STREAMLIT    │
                    │  🏠 Inicio · 📊 Stats · 🔮│
                    └─────────────────────────┘
```

**Variables de entrada:** `VIA_TRANSP`, `NITIMPOR`, `NANDINA`, `PAISORIGEN`, `PAISPROC`, `PAISCOMPRA`, `KILOSNETO`, `FOBDO`, `FLETES`

---

## 📁 Estructura del Proyecto

```
pharma-imports-ml/
├── app.py                      # 🏠 Entrada principal de Streamlit
├── utils.py                    # ⚙️  Carga de modelos, predicción, UI
├── script.py                   # 🔧 Pipeline de entrenamiento + evaluación
├── pages/
│   ├── pronosticos.py          # 🔮 Página de predicción
│   └── estadisticas.py         # 📊 Estadísticas y EDA
├── data/
│   └── data.csv                # Dataset fuente (no versionado)
├── modelos/
│   └── modelos_nandina/        # Modelos LightGBM serializados (no versionados)
├── media/                      # Recursos de la app
├── archivos_jupyter/           # Notebooks de EDA y validación
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Requisitos e Instalación

**Requisitos:** Python 3.10+

```bash
# 1. Clonar el repositorio
git clone https://github.com/JuanDiego076/pharma-imports-ml.git
cd pharma-imports-ml

# 2. Crear y activar un entorno virtual
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar el dataset en: data/data.csv
```

---

## 🚀 Cómo Ejecutarlo

```bash
# 1. Entrenar los modelos y ver las métricas de evaluación
python script.py

# 2. Lanzar la aplicación Streamlit
streamlit run app.py
```

La app abrirá en `http://localhost:8501`

---

## 🔑 Decisiones Técnicas Clave

**IDs y códigos como categóricos, no numéricos:** `NITIMPOR`, `NANDINA` y los códigos de país son identificadores — su valor numérico no es ordinal (un NIT de 860M no es "mayor" que uno de 800M). Tratarlos como categóricos en vez de numéricos fue el cambio que llevó la clasificación de un desempeño cercano al azar a 94%+ en el segmento Oncológico.

**Transformación logarítmica del precio:** la distribución de precios tiene outliers extremos (productos oncológicos cuestan órdenes de magnitud más que genéricos). Entrenar sobre `log1p(precio)` y revertir con `expm1` al evaluar estabiliza el modelo y rescató el segmento Farma de un R² negativo.

**Limpieza de precios inválidos:** se filtran registros con precio negativo (imposibles en una importación real) antes de entrenar — 57 filas en Farma. Datos sucios de entrada eran parte del problema, no solo el modelo.

**Segmentación por `TIPO_PRODUCTO`:** un modelo único mezclaría distribuciones de precio y demanda muy distintas. Modelos por segmento permiten aprender cada patrón de forma independiente.

**LightGBM sobre Random Forest:** entrena 3–10x más rápido en este volumen con desempeño comparable, clave para un bucle de entrenamiento multi-objetivo y multi-segmento.

---

## 📝 Aprendizajes

- El preprocesamiento importa más que el algoritmo: el mismo LightGBM pasó de ~2% a 94%+ de accuracy solo por tratar correctamente los tipos de variable.
- No todos los segmentos son iguales. Oncológico se predice muy bien; Farma es intrínsecamente difícil por su dispersión. Documentar el *porqué* de esa diferencia es parte del análisis.
- Validar siempre el rango de las predicciones: un modelo de regresión sin evaluación llegó a predecir precios de miles de millones negativos antes de corregir el tratamiento de variables.

---

## 👥 Autores

Proyecto final de **Análisis de Datos Innovador Avanzado — Talento Tech** (Grupo 518), Universidad de Antioquia, Noviembre 2025.

**Juan Diego Taborda Roldán** · [LinkedIn](https://www.linkedin.com/in/juandiegotabordaroldan) · juandiego276@gmail.com
Brayan Alexander Gallego · Liliana Córdoba Sinisterra · Byron Vanegas · Julio Correa · Carolina Yepes

---

<div align="center">
<sub>⭐ Si este proyecto te fue útil, considera darle una estrella.</sub>
</div>