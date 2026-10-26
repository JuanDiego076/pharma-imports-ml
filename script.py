import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import pickle
from pathlib import Path
from tqdm import tqdm
from itertools import islice


RUTA_MODELOS = Path("modelos/modelos_nandina")
RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
RUTA_SALIDA = Path("modelos/predicciones_nandina")
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# Clases con menos de este número de ejemplos se agrupan en "OTROS".
# Evita que el modelo intente aprender categorías con muy pocos datos.
MIN_EJEMPLOS_CLASE = 20


def optimizar_memoria(df):
    """Reduce la huella de memoria del DataFrame."""
    print(f"Memoria inicial: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'O':
            if df[col].nunique() < len(df) / 2:
                df[col] = df[col].astype('category')
        elif 'int' in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif 'float' in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], downcast='float')
    print(f"Memoria final: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df


def agrupar_clases_raras(serie, min_ejemplos=MIN_EJEMPLOS_CLASE):
    """Agrupa en 'OTROS' las categorías con menos de min_ejemplos apariciones."""
    conteo = serie.value_counts()
    clases_frecuentes = conteo[conteo >= min_ejemplos].index
    return serie.where(serie.isin(clases_frecuentes), other="OTROS")


# Cargar dataset
df = pd.read_csv("data/data.csv")
df = optimizar_memoria(df)

# Variables
targets_clas = ['principio activo', 'fabricante', 'marca']
target_reg = 'precio'

features = ['VIA_TRANSP', 'NITIMPOR', 'NANDINA', 'PAISORIGEN', 'PAISPROC',
            'PAISCOMPRA', 'KILOSNETO', 'FOBDO', 'FLETES']

# Identificadores y códigos: su valor numérico no es ordinal -> categóricos.
features_categoricas = ['VIA_TRANSP', 'NITIMPOR', 'NANDINA',
                        'PAISORIGEN', 'PAISPROC', 'PAISCOMPRA']


def entrenar_modelos(df, features, targets_clas, target_reg):

    for TIPO, grupo in tqdm(
        islice(df.groupby("TIPO_PRODUCTO"), 4),
        desc="Entrenando modelos..."):
        print(f"\n Entrenando modelo para TIPO_PRODUCTO = {TIPO} (n={len(grupo)})")

        grupo = grupo.dropna(subset=features + targets_clas + [target_reg]).copy()
        if grupo.empty:
            print(" Sin datos válidos, se omite este grupo.")
            continue

        # Un precio de importación no puede ser negativo: se filtran como inválidos.
        n_antes = len(grupo)
        grupo = grupo[grupo[target_reg] >= 0].copy()
        n_filtrados = n_antes - len(grupo)
        if n_filtrados > 0:
            print(f"  Filtradas {n_filtrados} filas con precio negativo (inválido)")

        X = grupo[features].copy()
        y_reg = grupo[target_reg].copy()

        # Agrupar clases raras en cada target de clasificación
        y_clas = pd.DataFrame(index=grupo.index)
        for col in targets_clas:
            y_clas[col] = agrupar_clases_raras(grupo[col].astype(str))

        le_dict = {}

        # Codificar features categóricas (identificadores y códigos)
        for col in features_categoricas:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le

        # Codificar targets de clasificación
        for col in targets_clas:
            le = LabelEncoder()
            y_clas[col] = le.fit_transform(y_clas[col].astype(str))
            le_dict[col] = le

        # Log-transform del precio para manejar outliers extremos
        y_reg_log = np.log1p(y_reg)

        # Dividir datos
        X_train, X_test, y_clas_train, y_clas_test, y_reg_train, y_reg_test = train_test_split(
            X, y_clas, y_reg_log, test_size=0.2, random_state=42
        )

        modelos_clas = {}

        print(f"\n  --- MÉTRICAS DE EVALUACIÓN ({TIPO}) ---")
        for target in targets_clas:
            n_clases = y_clas_train[target].nunique()
            lgbm_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
            lgbm_clf.fit(X_train, y_clas_train[target])
            modelos_clas[target] = lgbm_clf

            y_pred = lgbm_clf.predict(X_test)
            acc = accuracy_score(y_clas_test[target], y_pred)
            f1 = f1_score(y_clas_test[target], y_pred, average='weighted', zero_division=0)
            print(f"  [CLAS] {target:<18} accuracy={acc:.3f}  F1(weighted)={f1:.3f}  ({n_clases} clases)")

        # Regresor de precio (entrena en log, evalúa en pesos reales)
        lgbm_reg = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        lgbm_reg.fit(X_train, y_reg_train)

        y_pred_log = lgbm_reg.predict(X_test)
        y_pred_reg = np.expm1(y_pred_log)      # revertir a pesos reales
        y_reg_test_real = np.expm1(y_reg_test)  # valores reales en pesos

        mae = mean_absolute_error(y_reg_test_real, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_reg_test_real, y_pred_reg))
        r2 = r2_score(y_reg_test_real, y_pred_reg)
        baseline_pred = np.full_like(y_reg_test_real, y_reg_test_real.mean(), dtype=float)
        mae_baseline = mean_absolute_error(y_reg_test_real, baseline_pred)
        mejora = (1 - mae / mae_baseline) * 100
        print(f"  [REG]  precio             MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}")
        print(f"  [REG]  baseline (media)   MAE={mae_baseline:,.0f}  -> mejora: {mejora:.1f}%")

        data_modelo = {
            "modelo_clas": modelos_clas,
            "modelo_reg": lgbm_reg,
            "X_test": X_test,
            "y_clas_test": y_clas_test,
            "y_reg_test": y_reg_test,  # en escala log
            "label_encoders": le_dict,
            "features": features,
            "features_categoricas": features_categoricas,
            "targets_clas": targets_clas,
            "target_reg": target_reg,
            "precio_log": True,  # marca que el target está en log1p
        }

        with open(RUTA_MODELOS / f"modelo_{TIPO}.pkl", "wb") as f:
            pickle.dump(data_modelo, f)

        print(f"  Modelo guardado: modelo_{TIPO}.pkl")

    print("\nEntrenamiento finalizado.")


# Ejecutar
entrenar_modelos(df, features, targets_clas, target_reg)