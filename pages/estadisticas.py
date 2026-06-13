#Líbrerias
import streamlit as st
import utils as u
import pandas as pd
import numpy as np

#Barra lateral
u.generarMenu()

#titulo
st.title("Estadísticas del Proceso de Entrenamiento")

# Llamamos la función que muestra los datos
#modelo = u.ModeloImportaciones()

# Controles de usuario para nueva predicción
with st.sidebar:
    st.header("SEGMENTO")
    segmento = st.radio('Selecciona el tipo de producto:', ['Farma', 'ONCOLOGICO', 'PT'])
    st.header("Datos para la predicción")
    btn_ejecutar = st.button("Diagnosticar", type='primary')

if btn_ejecutar:
    
    u.generar_predicciones_por_nandina(segmento= segmento)