import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from components.model_def import make_bert

processed_data_path = 'data/processed/processed_data_bert.pkl'
model_save_path = 'saved_artifacts/model_bert_weights.h5'
model_export_dir = 'saved_artifacts/model_bert'

def train():
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)
        
        X_train_ids = data['X_train_bert_ids']
        X_train_mask = data['X_train_bert_mask']
        X_test_ids = data['X_test_bert_ids']
        X_test_mask = data['X_test_bert_mask']
        y_train = data['y_train']
        y_test = data['y_test']
    
    X_train = {'input_ids': X_train_ids, 'attention_mask': X_train_mask}
    X_test = {'input_ids': X_test_ids, 'attention_mask': X_test_mask}
    
    model = make_bert()
    
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=3, 
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop]
    )
    
    model.save_pretrained(model_export_dir)
    
#---
if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
        
