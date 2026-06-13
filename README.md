# 💊 Pharmaceutical Import Demand Forecasting
### Análisis Predictivo de Medicamentos Importados

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-9B59B6?style=for-the-badge&logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Production_Demo-27AE60?style=for-the-badge)](https://github.com/JuanDiego076/App-web-Analisis-de-importaciones)

---

## 🎯 Problem Statement

Pharmaceutical companies that import medications face a critical operational challenge: deciding **what to import, in what quantity, and when** — without visibility into future demand. Traditional approaches rely on historical intuition, creating two costly failure modes: **stockouts** (lost sales + patient risk) and **overstock** (storage costs + expiry risk).

This project transforms ~27,000 historical import transaction records into a **machine learning system** that predicts the active ingredient, manufacturer, brand, and price of future pharmaceutical imports — segmented by product type.

**Before this system:** Import decisions based on intuition and recent history, with no quantitative demand signal.  
**After this system:** LightGBM ensemble predicts import demand per pharmaceutical segment, delivered through a self-service Streamlit dashboard accessible to non-technical stakeholders.

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                 │
│  Raw Import Data          Preprocessing          Feature Store  │
│  (27,440 records)  ──►   LabelEncoding    ──►   8 Features     │
│  14 columns               Memory Optim.          Segmented      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    TRAINING LAYER        │
                    │  Segment: TIPO_PRODUCTO  │
                    │  ┌──────────────────┐   │
                    │  │ 3x LightGBM CLF  │   │  ← principio activo
                    │  │  (per segment)   │   │  ← fabricante
                    │  │                  │   │  ← marca
                    │  ├──────────────────┤   │
                    │  │ 1x LightGBM REG  │   │  ← precio
                    │  └──────────────────┘   │
                    │  Saved as: modelo_*.pkl  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   STREAMLIT DASHBOARD    │
                    │  ┌─────────────────────┐│
                    │  │  🏠 Home / Overview  ││
                    │  │  📊 Statistics       ││
                    │  │  🔮 Forecasting      ││
                    │  └─────────────────────┘│
                    └─────────────────────────┘
```

**Model input features:** `VIA_TRANSP`, `NITIMPOR`, `NANDINA`, `PAISORIGEN`, `PAISPROC`, `PAISCOMPRA`, `KILOSNETO`, `FOBDO`, `FLETES`

**Model outputs (per pharmaceutical segment):**
- `principio activo` — Active pharmaceutical ingredient (classification)
- `fabricante` — Manufacturer (classification)
- `marca` — Brand name (classification)
- `precio` — Import price in USD (regression)

---

## 📁 Project Structure

```
App-web-Analisis-de-importaciones/
├── app.py                      # 🏠 Main Streamlit entry point
├── utils.py                    # ⚙️  Model loading, prediction, UI helpers
├── script.py                   # 🔧 Model training pipeline
├── pages/
│   ├── pronosticos.py          # 🔮 Forecasting page (model inference)
│   └── estadisticas.py         # 📊 EDA & statistics page
├── data/
│   └── dataset_talento_tech.xlsx   # Source dataset (not tracked by git)
├── modelos/
│   └── modelos_nandina/        # Serialized LightGBM models (not tracked by git)
│       ├── modelo_ONCOLOGICO.pkl
│       ├── modelo_Farma.pkl
│       └── modelo_PT.pkl
├── media/                      # App assets (images)
├── archivos_jupyter/           # EDA notebooks (exploratory)
├── .streamlit/                 # Streamlit configuration
├── requirements.txt            # ← Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Prerequisites & Installation

**Requirements:**
- Python 3.10+
- pip

```bash
# 1. Clone the repository
git clone https://github.com/JuanDiego076/App-web-Analisis-de-importaciones.git
cd App-web-Analisis-de-importaciones

# 2. Create and activate a virtual environment
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the source dataset
# Place your dataset at: data/dataset_talento_tech.xlsx
```

---

## 🚀 How to Run

### Step 1 — Train the models

Run the training script first. This generates `.pkl` model files per pharmaceutical segment inside `modelos/modelos_nandina/`.

```bash
python script.py
```

Expected output:
```
Memoria inicial: ~X MB → Memoria final: ~Y MB
Entrenando modelos... 4/4 [========]
  Modelo Clasificación (principio activo) entrenado.
  Modelo Clasificación (fabricante) entrenado.
  Modelo Clasificación (marca) entrenado.
  Modelo Regresión (precio) entrenado.
Modelos guardados: modelo_ONCOLOGICO.pkl
...
Entrenamiento finalizado.
```

### Step 2 — Launch the Streamlit application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 27,440 import records |
| Pharmaceutical segments modeled | 4 (ONCOLOGICO, Farma, PT, + 1) |
| ML task type | Multi-target: 3 classifiers + 1 regressor per segment |
| Model architecture | LightGBM (classifier + regressor) |
| Predicted brand with highest demand | Keytruda 100mg (mk-3475): 1,400+ instances |
| Top import origin | China (~5,800 records), India (~3,000), USA (~1,700) |
| Primary transport mode | Maritime (~7,400 ops), Air (~5,300 ops) |

---

## 🔑 Key Technical Decisions

**LightGBM over Random Forest:** LightGBM trains 3–10x faster on this dataset size while achieving comparable or superior accuracy, critical for a multi-target, multi-segment training loop. `n_jobs=-1` enables parallel training across all available CPU cores.

**Segmentation by `TIPO_PRODUCTO`:** A single model for all pharmaceutical types would conflate wildly different price distributions and demand patterns (oncologics vs. generic farma). Segment-level models allow each group's patterns to be learned independently, confirmed by the high price variance in the EDA (`boxplot` shows extreme outliers driven by oncologic products like Keytruda).

**Memory optimization at load time:** `pd.to_numeric(downcast=...)` and `astype('category')` reduce DataFrame memory footprint before training, enabling the pipeline to run on memory-constrained environments without chunking.

**`LabelEncoder` persistence via pickle:** Encoders are saved alongside models to guarantee consistent label mapping at inference time — unseen values are handled with a fallback of `-1` rather than crashing.

---

## 📝 Lessons Learned

- **Multi-target classification requires N separate models with LightGBM** (no native multi-output support unlike sklearn's `MultiOutputClassifier`). Wrapping in a `{target: model}` dictionary keeps inference clean.
- A global price model with extreme outliers (Keytruda at 100x average price) strongly benefits from segment-level modeling before any normalization attempt.
- Streamlit's multi-page setup requires careful separation of session state between `app.py` (home) and individual pages — sharing model state through `utils.py` functions was the cleaner pattern.

---

## 📄 Full Technical Report

The complete EDA, methodology, and conclusions are documented in:
[`Análisis_predictivo_de_medicamentos_importados.pdf`](./Análisis_predictivo_de_medicamentos_importados.pdf)

---

## 👥 Authors

Developed as a final project for **Análisis de Datos Innovador Avanzado — Talento Tech** (Grupo 518) at Universidad de Antioquia, Noviembre 2025.

**Juan Diego Taborda Roldán** · [LinkedIn](https://www.linkedin.com/in/juandiegotabordaroldan) · [juandiego276@gmail.com](mailto:juandiego276@gmail.com)  
Brayan Alexander Gallego · Liliana Córdoba Sinisterra · Byron Vanegas · Julio Correa · Carolina Yepes

---

<div align="center">
<sub>⭐ If this project was useful to you, consider giving it a star.</sub>
</div>
