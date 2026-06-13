#Líbrerias
import streamlit as st
import utils as u
import pandas as pd
import numpy as np

#Barra lateral
u.generarMenu()

#titulo
st.title("Pronóstico Importaciones")

# Mostramos los datos
df_paises = pd.read_csv('data/paises.csv',sep=";")
dict_paises = dict(zip(df_paises['PAIS'],df_paises['CODIGO']))

# via transporte
df_via_transp = pd.read_csv('data/via_transporte.csv', sep=";")
dict_via_transp = dict(zip(df_via_transp['VIA_TRANSPORTE'],df_via_transp['CODIGO']))

ruta = 'data/dataset_talento_tech.xlsx'

# Llamamos la función que muestra los datos
modelo = u.ModeloImportaciones()
df = modelo.cargar_datos(ruta)

# Controles de usuario para nueva predicción
with st.sidebar:
    st.header("SEGMENTO")
    segmento = st.radio('Selecciona el tipo de producto:', ['Farma', 'ONCOLOGICO', 'PT'])
    st.header("Datos para la predicción")
    col1, col2 = st.columns(2)
    
    with col1:        
        VIA_TRANSP = st.selectbox('Vía de Transporte:', list(dict_via_transp.keys()))
        VIA_TRANSP = dict_via_transp[VIA_TRANSP]
        NITIMPOR = st.number_input("NIT Importador")
        NANDINA = st.number_input("Posición Arancelaria")
        PAISORIGEN = st.selectbox('País de origen:', list(dict_paises.keys()))
        PAISORIGEN = dict_paises[PAISORIGEN]
    with col2:
        PAISPROC = st.selectbox('País de procedencia:', list(dict_paises.keys()))
        PAISPROC = dict_paises[PAISPROC]
        PAISCOMPRA = st.selectbox('País de Compra:', list(dict_paises.keys()))
        PAISCOMPRA = dict_paises[PAISCOMPRA]
        KILOSNETO = st.number_input("Kilos Netos")
        FOBDO = st.number_input("Valor Importación")
        FLETES = st.number_input("Flete")
    
    col3, col4 = st.columns([3,2])
    
    with col4:
        btn_ejecutar = st.button("Diagnosticar", type='primary')

if btn_ejecutar:
    lista =  {
            'VIA_TRANSP':[VIA_TRANSP],
            'NITIMPOR':[NITIMPOR],
            'NANDINA':[NANDINA],
            'PAISORIGEN':[PAISORIGEN],
            'PAISPROC':[PAISPROC],
            'PAISCOMPRA':[PAISCOMPRA],
            'KILOSNETO':[KILOSNETO],
            'FOBDO':[FOBDO],
            'FLETES':[FLETES],
        }
    
    data = pd.DataFrame(lista)
    u.predecir(df=data, segmento= segmento)