import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_path = 'Datafiniti_Hotel_Reviews.csv'
processed_data_path = 'data/processed/processed_data.pkl'
vocab_size = 10000
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
    
    tokenizer = Tokenizer(num_words = vocab_size,
                          oov_token = '<OOV>')
    tokenizer.fit_on_texts(df['cleaned_text'])
    
    with open('saved_artifacts/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen = max_len)
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen = max_len)
    
    with open(processed_data_path, 'wb') as f:
        pickle.dump({
            'X_train_pad': X_train_pad,
            'X_test_pad': X_test_pad,
            'y_train': y_train.values,
            'y_test': y_test.values
        }, f)
        
#---
if __name__ == '__main__':
    prepare_data()
        
    
    
