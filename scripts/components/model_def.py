import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertConfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D


vocab_size = 10000
max_len = 150
embedding_dim = 128

def make_lstm():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length = max_len),
        SpatialDropout1D(0.2),
        LSTM(100),
        Dense(1, activation = 'sigmoid')
    ])
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    model.summary()
    return model

def make_stacked_lstm():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length = max_len),
        SpatialDropout1D(0.4),
        LSTM(units = 128, dropout = 0.2),
        Dense(1, activation = 'sigmoid')
    ])
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    model.summary()
    return model

def make_bert():
    config = BertConfig.from_pretrained('prajjwal1/bert-tiny', num_labels=2)
    model = TFBertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny',
                                                            config=config, from_pt=True)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
    )
    model.summary()
    return model














