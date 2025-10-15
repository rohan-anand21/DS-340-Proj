import pickle
import sys
import types
from fastapi.templating import Jinja2Templates
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import InputLayer, Embedding
from tensorflow.keras.utils import get_custom_objects
from .models.lstm import LSTM
from .models.stacked_lstm import StackedLSTM
from .models.bert import BERT
from tensorflow.keras.preprocessing.text import Tokenizer

legacy_module = types.ModuleType("keras.src.legacy.preprocessing.text")
legacy_module.Tokenizer = Tokenizer
legacy_preprocessing = types.ModuleType("keras.src.legacy.preprocessing")
legacy_preprocessing.text = legacy_module
legacy_legacy = types.ModuleType("keras.src.legacy")
legacy_legacy.preprocessing = legacy_preprocessing
legacy_src = types.ModuleType("keras.src")
legacy_src.legacy = legacy_legacy

sys.modules.setdefault("keras.src", legacy_src)
sys.modules.setdefault("keras.src.legacy", legacy_legacy)
sys.modules.setdefault("keras.src.legacy.preprocessing", legacy_preprocessing)
sys.modules.setdefault("keras.src.legacy.preprocessing.text", legacy_module)

_original_from_config = InputLayer.from_config.__func__

def _patched_from_config(cls, config):
    config = dict(config)
    if 'batch_shape' in config and 'batch_input_shape' not in config:
        config['batch_input_shape'] = config.pop('batch_shape')
    return _original_from_config(cls, config)

InputLayer.from_config = classmethod(_patched_from_config)

_embedding_from_config = Embedding.from_config.__func__

def _patched_embedding_from_config(cls, config):
    config = dict(config)
    dtype_cfg = config.get('dtype')
    if isinstance(dtype_cfg, dict) and dtype_cfg.get('class_name') == 'DTypePolicy':
        config['dtype'] = dtype_cfg.get('config', {}).get('name', 'float32')
    return _embedding_from_config(cls, config)

Embedding.from_config = classmethod(_patched_embedding_from_config)

class _LegacyDTypePolicy:
    def __init__(self, name):
        self.name = name
    @property
    def compute_dtype(self):
        return self.name
    @property
    def variable_dtype(self):
        return self.name

get_custom_objects().setdefault('DTypePolicy', _LegacyDTypePolicy)

MAX_LEN = 150

print("Loading all artifacts...")
with open("saved_artifacts/tokenizer.pkl", 'rb') as f:
    keras_tokenizer = pickle.load(f)

simple_lstm_model = LSTM()
stacked_lstm_model = StackedLSTM()
bert_model = BERT(model_path="saved_artifacts/model_bert")
print("Artifacts loaded.")

def preprocess_for_lstm(text: str):
    sequence = keras_tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_LEN)

templates = Jinja2Templates(directory="app/templates")

models_and_helpers = {
    "simple_lstm": simple_lstm_model,
    "stacked_lstm": stacked_lstm_model,
    "bert": bert_model,
    "preprocess_lstm": preprocess_for_lstm
}

async def get_models():
    return models_and_helpers
