import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from components.model_def import make_stacked_lstm

processed_data_path = 'data/processed/processed_data.pkl'
model_save_path = 'saved_artifacts/model_lstm_stacked.h5'

def train():
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)
        X_train_pad = data['X_train_pad']
        X_test_pad = data['X_test_pad']
        y_train = data['y_train']
        y_test = data['y_test']

    model = make_stacked_lstm()

    checkpoint = ModelCheckpoint(model_save_path, 
                                 save_best_only=True, 
                                 monitor='val_accuracy', 
                                 mode='max')
    early_stop = EarlyStopping(monitor='val_loss', 
                               patience=3, 
                               restore_best_weights=True)

    model.fit(
        X_train_pad, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test_pad, y_test),
        callbacks=[checkpoint, early_stop]
    )

if __name__ == '__main__':
    train()
