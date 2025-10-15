import pandas as pd
import re
from sklearn.model_selection import train_test_split
import pickle
from transformers import BertTokenizer

data_path = 'Datafiniti_Hotel_Reviews.csv'
processed_data_path = 'data/processed/processed_data_bert.pkl'
max_len = 150

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def prepare_data():
    df = pd.read_csv(data_path)
    df = df[['reviews.rating', 'reviews.text']]
    df.dropna(inplace = True)
    
    df = df[(df['reviews.rating'] <= 2) | (df['reviews.rating'] >= 4)]
    df['label'] = (df['reviews.rating'] >= 4).astype(int)
    df['cleaned_text'] = df['reviews.text'].apply(clean_text)
    
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    
    X_train_bert = tokenizer(
        text = X_train.tolist(),
        add_special_tokens = True,
        max_length = max_len,
        truncation = True,
        padding = 'max_length',
        return_tensors = 'tf',
        return_token_type_ids = False,
        return_attention_mask = True
    )
    
    X_test_bert = tokenizer(
        text=X_test.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True
    )
    
    with open(processed_data_path, 'wb') as f:
        pickle.dump({
            'X_train_bert_ids': X_train_bert['input_ids'],
            'X_train_bert_mask': X_train_bert['attention_mask'],
            'X_test_bert_ids': X_test_bert['input_ids'],
            'X_test_bert_mask': X_test_bert['attention_mask'],
            'y_train': y_train.values,
            'y_test': y_test.values
        }, f)

#---
if __name__ == '__main__':
    prepare_data()
    
    
    
