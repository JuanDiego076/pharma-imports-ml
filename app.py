import enum
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import utils as u
import media as md
from PIL import Image

# Configuramos la página
st.set_page_config(page_title="Importaciones de Productos Farmacéuticos",
                   page_icon="💊",
                   layout="wide"
                   )

# Variables
ruta = ""
mod = u.ModeloImportaciones()

# Interfaz

#st.title("📊 Predicción de Importaciones de Productos Farmacéuticos")

u.generarMenu()

# modificaciones
# creamos columnas
col1, col2, col3 = st.columns([1,4,1], vertical_alignment='center')
with col2:
    
    st.title('📊 Predicción de Importaciones de Productos Farmacéuticos')
    #st.subheader("Predicción de enfermedad Cardiovascular")
    st.write("""
            Análisis predictivo para la optimización de importaciones farmacéuticas, 
            diseñado para enfrentar los desafíos que presenta la farmacéutica en la gestión de la cadena de suministro internacional.
            Mejorando la planificación de importación para predecir los medicamentos que tendrán mayor demanda futura, 
            por medio de la identificación de patrones y tendencias que orienten la toma de decisiones estratégicas en la empresa con el fin de prevenir riesgos de desabastecimiento en medicamentos de alta demanda o sobre-inventario usando el modelo de lightgbm.
            """)
    logo = Image.open("media/Importaciones.jpg")
    st.image(logo)
    
    
    default_path = f"{ruta}data/dataset_talento_tech.xlsx"
    
    if os.path.exists(default_path):
        st.info(f"Datos Originales")
        df = mod.cargar_datos(default_path)
        st.dataframe( df.head(5))
        
   
# Footer
hide_streamlit_style = """
            <style>
            foorter {visibility: hidden;}
            </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# cargar dataset 
def datos():
    with st.sidebar:
        default_path = f"{ruta}data/dataset_talento_tech.xlsx"
        archivo = st.file_uploader("Sube tu dataset (.xlsx)", type=["xlsx"])

        if archivo is not None:
            df = mod.cargar_datos(archivo)
        else:
            if os.path.exists(default_path):
                st.info(f"Usando archivo por defecto: `{default_path}`")
                df = mod.cargar_datos(default_path)
                return df
            else:
                st.error("No se ha subido ninggún archivo y no existe archivo por defecto")
                st.stop()
                return pd.DataFrame()



