from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

model_path = "./saved_model" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

username = "chemsdine"
password = "root"
host = "stackoverflow.mysql.database.azure.com"
port = 3306
dbname = "stackoverflow"
connection_string = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_string)

class UserInput(BaseModel):
    title: str
    question: str

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words if word != 'p']
    return ' '.join(words)

def predict_tags(text):
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_tags = logits.argmax(dim=1)
    return predicted_tags.tolist()

@app.post("/predict")
def predict(input: UserInput):
    combined_text = input.title + " " + input.question
    prediction = predict_tags(combined_text)
    prediction = prediction.tolist() 

    df = pd.DataFrame([[input.title, input.question, prediction]], columns=['title', 'question', 'prediction'])
    df.to_sql(name='input_table', con=engine, if_exists='replace', index=False)

    return {"prediction": prediction}
