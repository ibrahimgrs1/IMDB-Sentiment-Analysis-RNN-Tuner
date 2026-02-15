
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.datasets import imdb  # veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, roc_curve, auc

import kerastuner as kt
from kerastuner.tuners import RandomSearch

import warnings
warnings.filterwarnings("ignore")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train, 100)
x_test = pad_sequences(x_test, 100)


def build_model(hp):
    
    model = Sequential()
    
    model.add(Embedding(
        input_dim=10000,
        output_dim=hp.Int("embedding_out", min_value=32, max_value=128, step=32), # Virgül eklendi
        input_length=100
    ))
    model.add(SimpleRNN(
        units=hp.Int("rnn_units", min_value=32, max_value=128, step=32)
    ))
    
    model.add(Dropout(rate = hp.Float("rate", min_value = 0.2, max_value = 0.6, step = 0.1)))
    
    model.add(Dense(1, activation = "sigmoid"))
    
    
    model.compile(optimizer = hp.Choice("optimizer" , ["adam", "rmsprop"]),
                  loss = "binary_crossentropy",
                  metrics = ["accuracy", tf.keras.metrics.AUC(name='auc')])
    return model
    
    
tuner = RandomSearch(build_model,
                     objective = "val_loss",
                     max_trials = 4,
                     executions_per_trial= 1,
                     directory = "rnn_tuner_directory",
                     project_name = "imdb_run"
                     )    

earlystopping = EarlyStopping(monitor = "loss", patience = 5, restore_best_weights = True)

    
tuner.search(x_train, y_train,
             epochs = 10,
             validation_split = 0.2,
             callbacks = [earlystopping]
             )

best_model = tuner.get_best_models(num_models = 1)[0]
best_model.save("imdb_rnn_best_model.keras")

loss, accuracy, auc_score = best_model.evaluate(x_test, y_test)
print(f"\nTest Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
print(f"Test AUC: {auc_score}")