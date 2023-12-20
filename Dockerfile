# On utilise l'image de base Python
FROM python:3.8-slim-buster

# On spécifie le chemin ou on souhaite copier l'app dans le conteneur.
WORKDIR /app

# Exécution du fichier requirements pour installer tous les dépendances nécéssaire.
COPY requirements.txt .

# Installation des dépendences
RUN pip install --no-cache-dir -r requirements.txt

# On copie le fichier principale d'éxecution du streamlit dans le conteneur.
COPY api.py .

COPY Streamlit/ model_app.py
COPY Streamlit/ Stack.png
COPY Streamlit/ style.css

COPY Traitement/ ingest.py
COPY Traitement/ model.py
COPY Traitement/ preprocess.py

# On spécifie le port.
EXPOSE 8501

# On spécifie la commande à saisir pour exécuter l'app.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]

# execution API : docker run -p 80:80 stackoverflow:v1