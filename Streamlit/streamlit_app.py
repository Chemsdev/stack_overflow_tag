import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import numpy as np
import module_app
from PIL import Image
import io



# Set le titre de la page
st.set_page_config(page_title='My Streamlit App')

#Import du CSS
module_app.local_css("style.css")

def  page1():
    st.title("PAGE 1")
    
def  page2():
    st.title("PAGE 2")
  
def  page3():
    st.title("PAGE 3")
    
def accueil():

    st.title("ACCUEIL")
   
#---------------------  Sidebar  ----------------------#
# Menu déroulant pour sélectionner la page à afficher
menu = ["ACCEUIL", "PAGE 1", "PAGE 2", "PAGE 3"]
choice = st.sidebar.selectbox(" ", menu)
st.sidebar.title("STACK OVERFLOW")
# Chargement de l'image à partir d'un fichier local
image_path = "Stack.png"
image = Image.open(image_path)

# Redimensionnement de l'image
width, height = image.size
new_width = int(width * 0.9)  # Réduire la largeur de l'image de moitié
new_height = int(height * 0.9)  # Réduire la hauteur de l'image de moitié
resized_image = image.resize((new_width, new_height))

# Conversion de l'image redimensionnée en format PNG
image_io = io.BytesIO()
resized_image.save(image_io, format='PNG')

# Affichage de l'image redimensionnée dans Streamlit
st.sidebar.image(image_io, caption='Image de description', use_column_width=True)


# Affichage de la page correspondant à la sélection du menu
if choice == "Page1":
    page1()
elif choice == "Page2":
    page2()
elif choice == "Page3":
    page3()
else:
    accueil()
