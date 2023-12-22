import numpy as np
import pandas as pd
import pymysql
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import TrainingArguments, Trainer

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

query = "SELECT * FROM stackoverflow.input_table"

df = pd.read_sql_query(query, connection)

connection.close()

# Train the model
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
sentences = df['Combined'].tolist()
tags = df['Tags'].tolist()

# Split the tags into a list of tags for each example
tag_lists = [tags.split('<>') for tags in tags]

all_tags = [tag for tag_list in tags for tag in tags]

unique_tags = list(set(all_tags))

num_labels = len(unique_tags)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

labels = []
for tag in tags:
    tag_list = tag.split('<>')
    label = [1 if unique_tag in tag_list else 0 for unique_tag in unique_tags]
    labels.append(label)
labels = torch.tensor(labels, dtype=torch.float32)

input_ids = tokenized_inputs['input_ids']
attention_masks = tokenized_inputs['attention_mask']
if 'token_type_ids' in tokenized_inputs:
    token_type_ids = tokenized_inputs['token_type_ids']
else:
    token_type_ids = None

assert len(labels) == len(input_ids) == len(attention_masks), "y a un mismatch dans le nombre de sample entre le label et les input tokenized"

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.1, random_state=42)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.1, random_state=42)

if token_type_ids is not None:
    train_type_ids, validation_type_ids, _, _ = train_test_split(token_type_ids, labels, test_size=0.1, random_state=42)

train_dataset = TextDataset({'input_ids': train_inputs, 'attention_mask': train_masks}, train_labels)
val_dataset = TextDataset({'input_ids': validation_inputs, 'attention_mask': validation_masks}, validation_labels)

training_args = TrainingArguments(
    per_device_train_batch_size=2,   
    per_device_eval_batch_size=2,    
    learning_rate=3e-5,              
    output_dir="./output",
    evaluation_strategy="epoch",
    num_train_epochs=1,              
    save_total_limit=1,
    save_strategy="no",             
    logging_dir="./logs",
    logging_steps=2000,              
    load_best_model_at_end=False,
    metric_for_best_model='mse',
)

train_size = int(0.1 * len(train_dataset))
small_train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()