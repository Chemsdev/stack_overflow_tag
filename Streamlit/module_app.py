import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Fonction qui charge le fichier css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


    