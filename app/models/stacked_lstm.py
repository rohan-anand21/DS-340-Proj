from tensorflow.keras.models import load_model

class StackedLSTM:
    def __init__(self, model_path = 'saved_artifacts/model_lstm_stacked.h5'):
        self.model = load_model(model_path, compile=False)
        
    def predict(self, preprocessed_text):
        prob = self.model.predict(preprocessed_text)[0][0]
        return "Positive" if prob > 0.5 else "Negative"
