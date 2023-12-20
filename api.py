import json
# import mysql.connector
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# Configuration MySQL
# mysql_host = "YOUR_MYSQL_HOST"
# mysql_user = "YOUR_MYSQL_USER"
# mysql_password = "YOUR_MYSQL_PASSWORD"
# mysql_database = "YOUR_MYSQL_DATABASE"

app = Flask(__name__)

# Fonction pour établir une connexion à la base de données MySQL
# def get_mysql_connection():
#     return mysql.connector.connect(
#         host=mysql_host,
#         user=mysql_user,
#         password=mysql_password,
#         database=mysql_database
# )

@app.route('/')
def index():
    return "Hello World!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         title = data.get("title", "")
#         question = data.get("question", "")

#         # Prédiction (à adapter selon votre modèle)
#         tags = predict_tags(title, question)

#         # Stockage dans la base de données MySQL
#         connection = get_mysql_connection()
#         cursor = connection.cursor()

#         # Insérer les prédictions dans une table dédiée
#         insert_query = "INSERT INTO Predictions (title, question, tags) VALUES (%s, %s, %s)"
#         cursor.execute(insert_query, (title, question, json.dumps(tags)))
#         connection.commit()
#         connection.close()

#         # Retourne les tags prédits à Streamlit
#         return jsonify({"tags": tags})

#     except Exception as e:
#         return jsonify({"error": str(e)})

# def predict_tags(title, question):
#     tags = "task 1"
#     return list(tags)

# def pre process
if __name__ == '__main__':
    app.run(debug=True)
