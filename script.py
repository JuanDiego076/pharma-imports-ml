import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import pickle
from pathlib import Path
from tqdm import tqdm
from itertools import islice


RUTA_MODELOS = Path("modelos/modelos_nandina")
RUTA_MODELOS.mkdir(exist_ok=True)
RUTA_SALIDA = Path("modelos/predicciones_nandina")
RUTA_SALIDA.mkdir(exist_ok=True)



def optimizar_memoria(df):
    """Reduce la huella de memoria del DataFrame."""
    print(f"Memoria inicial: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    for col in df.columns:
        # Optimizar categóricas/object (lo más importante)
        if df[col].dtype == 'object' or df[col].dtype == 'O':
            # Asegurar que no sean demasiados valores únicos antes de convertir a category
            if df[col].nunique() < len(df) / 2: 
                df[col] = df[col].astype('category')
        # Optimizar numéricas
        elif 'int' in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif 'float' in str(df[col].dtype):
            df[col] = pd.to_numeric(df[col], downcast='float')
    print(f"Memoria final: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

# Cargar dataset
df = pd.read_excel("data/dataset_talento_tech.xlsx")

# Aplicar al inicio
df = optimizar_memoria(df)

# Variables
targets_clas = ['principio activo', 'fabricante', 'marca']
target_reg = 'precio'

features = ['VIA_TRANSP', 'NITIMPOR','NANDINA','PAISORIGEN','PAISPROC'
                                           ,'PAISCOMPRA','KILOSNETO','FOBDO', 'FLETES']




def entrenar_modelos(df, features, targets_clas, target_reg):
    
    for TIPO, grupo in tqdm(
        islice(df.groupby("TIPO_PRODUCTO"),4), 
        desc="Entrenando modelos..."):
        print(f"\n Entrenando modelo para NANDINA = {TIPO} (n={len(grupo)})")
    
        grupo = grupo.dropna(subset=features + targets_clas + [target_reg])
        if grupo.empty:
            print(" Sin datos válidos, se omite este grupo.")
            continue

        X = grupo[features].copy()
        y_clas = grupo[targets_clas].copy()
        y_reg = grupo[target_reg].copy()

        # Codificar variables categóricas
        le_dict = {}        
        for col in X.select_dtypes(include=["object",'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

        for col in y_clas.select_dtypes(include=["object",'category']).columns:
            le = LabelEncoder()
            y_clas[col] = le.fit_transform(y_clas[col].astype(str))
            le_dict[col] = le

        # Dividir datos
        X_train, X_test, y_clas_train, y_clas_test, y_reg_train, y_reg_test = train_test_split(
            X, y_clas, y_reg, test_size=0.2, random_state=42
        )
        
        modelos_clas = {}
        # LGBM no maneja Multi-target Classification nativamente, entrenamos 3 modelos:
        for target in targets_clas:
            # Usamos n_jobs=-1, pero LGBM es eficiente, puedes probarlo.
            lgbm_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1) 
            lgbm_clf.fit(X_train, y_clas_train[target])
            modelos_clas[target] = lgbm_clf
            print(f"  Modelo Clasificación ({target}) entrenado.")
        
        lgbm_reg = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        lgbm_reg.fit(X_train, y_reg_train)
        print("  Modelo Regresión (precio) entrenado.")
        

        # Guardar
        data_modelo = {
            #"modelo_clas": clf,
            "modelo_clas": modelos_clas,
            #"modelo_reg": reg,
            "modelo_reg": lgbm_reg,
            "X_test":X_test,
            "label_encoders": le_dict,
            "features": features,
            "targets_clas": targets_clas,
            "target_reg": target_reg,
        }

        with open(RUTA_MODELOS / f"modelo_{TIPO}.pkl", "wb") as f:
            pickle.dump(data_modelo, f)

        print(f"Modelos guardados: modelo_{TIPO}.pkl")

    print("\nEntrenamiento finalizado.")


def generar_predicciones_por_nandina(df):
    for nandina, grupo in tqdm(
        islice(df.groupby("TIPO_PRODUCTO"), 4), 
        desc="Entrenando modelos..."):
        modelo_path = RUTA_MODELOS / f"modelo_{nandina}.pkl"
        if not modelo_path.exists():
            print(f" No se encontró modelo para NANDINA = {nandina}. Se omite.")
            continue
        print(f" Se encontró modelo para NANDINA = {nandina}. --- {modelo_path}")
        with open(modelo_path, "rb") as f:
            data_modelo = pickle.load(f)

        modelos_clas = data_modelo["modelo_clas"] # Cargar diccionario
        modelo_reg = data_modelo["modelo_reg"]
        le_dict = data_modelo["label_encoders"]
        features = data_modelo["features"]
        targets_clas = data_modelo["targets_clas"]
        target_reg = data_modelo["target_reg"]

        X = grupo[features].copy()
        print(X.head())
        for col in X.select_dtypes(include=["object",'category']).columns:
            if col in le_dict:
                le = le_dict[col]
                X[col] = X[col].map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)
        
        df_pred_clas = pd.DataFrame(index=grupo.index)
        # Iterar sobre los modelos de clasificación
        print (f"Modelos de clasificacion {modelos_clas}")
        for target, modelo_clas in modelos_clas.items():
            y_pred_clas = modelo_clas.predict(X)
            df_pred_clas[target] = y_pred_clas
            
        y_pred_reg = modelo_reg.predict(X)
        df_pred_reg = pd.DataFrame({target_reg: y_pred_reg}, index=grupo.index)

        # Decodificar clasificaciones
        for col in df_pred_clas.columns:
            if col in le_dict:
                le = le_dict[col]
                df_pred_clas[col] = le.inverse_transform(df_pred_clas[col].astype(int))

        df_result = pd.concat(
            [grupo.reset_index(drop=True), df_pred_clas, df_pred_reg], axis=1
        )

        archivo_salida = RUTA_SALIDA / f"predicciones_{nandina}.pkl"
        df_result.to_pickle(archivo_salida)
        print(f"Predicciones guardadas en: {archivo_salida}")

    print("\nProceso de predicciones completado.")

# Ejecutar
entrenar_modelos(df, features, targets_clas, target_reg)
#generar_predicciones_por_nandina(df)