import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification

class BERT:
    def __init__(self, model_path = 'saved_artifacts/model_bert'):
        self.model = TFBertForSequenceClassification.from_pretrained(model_path)
        
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        self.max_len = 150

    def predict(self, preprocessed_text):
        inputs = self.tokenizer(
            text=preprocessed_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='tf',
            return_attention_mask=True
        )
        
        model_output = self.model({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })
        
        logits = model_output.logits[0]
        probabilities = tf.nn.softmax(logits).numpy()
        predicted_class_id = np.argmax(probabilities)
        
        return "Positive" if predicted_class_id == 1 else "Negative"
        
