import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pickle import dump, load
import pandas as pd
import os
import lightgbm

ruta = r"media\corazon.png"

# Función para la barra lateral sidebar
def generarMenu():
    with st.sidebar:
        st.header("Menú")
        # mostramos una lista personalizadas de mis páginas
        st.page_link('app.py', label='Inicio', icon='🏠')
        st.page_link('pages/pronosticos.py', label='Pronóstico')
        st.page_link('pages/estadisticas.py', label='Estadísticas', icon='📊')

# Creamos la clase ModeloImportaciones
class ModeloImportaciones():
    # Constructor
    def __init__(self):
        self.df = None
        self.pred = None
        
    def cargar_datos(self, archivo):
        df = pd.read_excel(archivo)
        self.df = df
        return df
    
    # Cargar predicciones
    def cargar_predicciones (self, ruta, modelo):
        if os.path.exists(ruta):
            df_pred = pd.read_pickle(ruta)

            if self.df is not None:
                df_origen = self.df.copy()
                # Elimino duplicados de ciudad y país en el dataset original
                df_origen = df_origen[["Ciudad", "País"]].drop_duplicates()
                # Unimos df_pred con las columnas del df_origen
                df_pred = df_pred.merge(df_origen, on="Ciudad", how="left") 

            self.df_pred = df_pred
            return df_pred
        return None
    
def generar_predicciones_por_nandina(segmento):
    modelo_path =  f"modelos/modelos_nandina/modelo_{segmento}.pkl"
    #if not modelo_path.exists():
    #    st.write(f" No se encontró modelo para NANDINA = {segmento}. Se omite.")
        
    st.write(f" Se encontró modelo para NANDINA = {segmento}. --- {modelo_path}")
    data_modelo = load(open(modelo_path, "rb")) 

    modelos_clas = data_modelo["modelo_clas"] # Cargar diccionario
    modelo_reg = data_modelo["modelo_reg"]
    le_dict = data_modelo["label_encoders"]
    features = data_modelo["features"]
    targets_clas = data_modelo["targets_clas"]
    target_reg = data_modelo["target_reg"]
    X_test = data_modelo["X_test"]

    X = X_test[features].copy()
    for col in X.select_dtypes(include=["object",'category']).columns:
        if col in le_dict:
            le = le_dict[col]
            X[col] = X[col].map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)
            
    # --- Predicciones LGBM ---
    
    df_pred_clas = pd.DataFrame(index=X_test.index)
    # Iterar sobre los modelos de clasificación
    print (f"Modelos de clasificacion {modelos_clas}")
    for target, modelo_clas in modelos_clas.items():
        y_pred_clas = modelo_clas.predict(X)
        df_pred_clas[target] = y_pred_clas
        
    y_pred_reg = modelo_reg.predict(X)
    df_pred_reg = pd.DataFrame({target_reg: y_pred_reg}, index=X_test.index)

    # Decodificar clasificaciones
    for col in df_pred_clas.columns:
        if col in le_dict:
            le = le_dict[col]
            df_pred_clas[col] = le.inverse_transform(df_pred_clas[col].astype(int))

    df_result = pd.concat(
        [X_test.reset_index(drop=True), df_pred_clas, df_pred_reg], axis=1
    )
    
    df_result = df_result[~(df_result['principio activo'].isna()) & ~(df_result['NANDINA'].isna())]
    st.subheader(f"Producto {segmento}")
    st.markdown("- Datos estadísticos del Entrenamiento")
    st.dataframe( df_result)
        
    
def predecir (df, segmento):
    
    if segmento == 'ONCOLOGICO':
        modelo_cargado = load(open("modelos/modelos_nandina/modelo_ONCOLOGICO.pkl", "rb"))        
        modelos_clas = modelo_cargado["modelo_clas"] # Cargar diccionario
        modelo_reg = modelo_cargado["modelo_reg"]
        le_dict = modelo_cargado["label_encoders"]
        features = modelo_cargado["features"]
        targets_clas = modelo_cargado["targets_clas"]
        target_reg = modelo_cargado["target_reg"]
        X = df[features].copy()
        
        for col in X.select_dtypes(include=["object",'category']).columns:
            if col in le_dict:
                le = le_dict[col]
                X[col] = X[col].map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

        # --- Predicciones LGBM ---
        
        df_pred_clas = pd.DataFrame(index=df.index)
        # Iterar sobre los modelos de clasificación
        for target, modelo_clas in modelos_clas.items():
            y_pred_clas = modelo_clas.predict(X)
            df_pred_clas[target] = y_pred_clas
            
        y_pred_reg = modelo_reg.predict(X)
        df_pred_reg = pd.DataFrame({target_reg: y_pred_reg}, index=df.index)

        # Decodificar clasificaciones
        for col in df_pred_clas.columns:
            if col in le_dict:
                le = le_dict[col]
                df_pred_clas[col] = le.inverse_transform(df_pred_clas[col].astype(int))
        df_result = pd.concat(
            [df.reset_index(drop=True), df_pred_clas, df_pred_reg], axis=1
        )
        st.subheader("Producto Oncológico")
        st.markdown("- Datos Pronósticados")
        st.dataframe( df_result)
        
        
        
        
    elif segmento == 'Farma':
        modelo_cargado = load(open("modelos/modelos_nandina/modelo_Farma.pkl", "rb"))        
        modelos_clas = modelo_cargado["modelo_clas"] # Cargar diccionario
        modelo_reg = modelo_cargado["modelo_reg"]
        le_dict = modelo_cargado["label_encoders"]
        features = modelo_cargado["features"]
        targets_clas = modelo_cargado["targets_clas"]
        target_reg = modelo_cargado["target_reg"]
        X = df[features].copy()
        
        for col in X.select_dtypes(include=["object",'category']).columns:
            if col in le_dict:
                le = le_dict[col]
                X[col] = X[col].map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

        # --- Predicciones LGBM ---
        
        df_pred_clas = pd.DataFrame(index=df.index)
        # Iterar sobre los modelos de clasificación
        for target, modelo_clas in modelos_clas.items():
            y_pred_clas = modelo_clas.predict(X)
            df_pred_clas[target] = y_pred_clas
            
        y_pred_reg = modelo_reg.predict(X)
        df_pred_reg = pd.DataFrame({target_reg: y_pred_reg}, index=df.index)

        # Decodificar clasificaciones
        for col in df_pred_clas.columns:
            if col in le_dict:
                le = le_dict[col]
                df_pred_clas[col] = le.inverse_transform(df_pred_clas[col].astype(int))
        df_result = pd.concat(
            [df.reset_index(drop=True), df_pred_clas, df_pred_reg], axis=1
        )
        st.subheader("Producto Farma")
        st.markdown("- Datos Pronósticados")
        st.dataframe( df_result)
         
    elif segmento == 'PT':
        modelo_cargado = load(open("modelos/modelos_nandina/modelo_PT.pkl", "rb"))        
        modelos_clas = modelo_cargado["modelo_clas"] # Cargar diccionario
        modelo_reg = modelo_cargado["modelo_reg"]
        le_dict = modelo_cargado["label_encoders"]
        features = modelo_cargado["features"]
        targets_clas = modelo_cargado["targets_clas"]
        target_reg = modelo_cargado["target_reg"]
        X = df[features].copy()
        
        for col in X.select_dtypes(include=["object",'category']).columns:
            if col in le_dict:
                le = le_dict[col]
                X[col] = X[col].map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

        # --- Predicciones LGBM ---
        
        df_pred_clas = pd.DataFrame(index=df.index)
        # Iterar sobre los modelos de clasificación
        for target, modelo_clas in modelos_clas.items():
            y_pred_clas = modelo_clas.predict(X)
            df_pred_clas[target] = y_pred_clas
            
        y_pred_reg = modelo_reg.predict(X)
        df_pred_reg = pd.DataFrame({target_reg: y_pred_reg}, index=df.index)

        # Decodificar clasificaciones
        for col in df_pred_clas.columns:
            if col in le_dict:
                le = le_dict[col]
                df_pred_clas[col] = le.inverse_transform(df_pred_clas[col].astype(int))
        df_result = pd.concat(
            [df.reset_index(drop=True), df_pred_clas, df_pred_reg], axis=1
        )
        st.subheader("Producto Terminado")
        st.markdown("- Datos Pronósticados")
        st.dataframe( df_result)