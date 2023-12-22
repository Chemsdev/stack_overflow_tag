import numpy as np
import pandas as pd
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt')
nltk.download('stopwords')

# connect to the db
username = "chemsdine"
password = "root"
host = "stackoverflow.mysql.database.azure.com"
port = 3306
dbname = "stackoverflow"

connection = pymysql.connect(
    host=host,
    user=username,
    password=password,
    database=dbname,
    port=port,
    cursorclass=pymysql.cursors.DictCursor 
)

query = "SELECT * FROM stackoverflow.raw"

df = pd.read_sql_query(query, connection)

connection.close()


# preprocess
def preprocess_text(text):
    words = word_tokenize(text.lower())

    words = [word for word in words if word.isalpha()]

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    lemmatizer = nltk.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    words = [word for word in words if word != 'p']

    return ' '.join(words)

df['Combined'] = df['Title'] + ' ' + df['Body']
df['Combined'] = df['Combined'].apply(preprocess_text)
df['Tags'] = df['Tags'].apply(preprocess_text)

input_df = df[['Combined', 'Tags']]

connection_string = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_string)

table_name = 'input_table'
input_df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)