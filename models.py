import tensorflow as tf
from keras.models import Model
from keras import Input
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from dictionaries import classes, symbol

def motor(
        n_neurons_lstm,
        n_neurons_timedistributed,
        learning_rate,
):
    inputs = Input(shape=(None,))

    embeddings = Embedding(input_dim=len(symbol),
                           output_dim=25)(inputs)

    blstm1 = Bidirectional(LSTM(units=n_neurons_lstm,
                                     return_sequences=True))(embeddings)
    dropout1 = Dropout(0.5)(blstm1)
    blstm2 = Bidirectional(LSTM(units=n_neurons_lstm,
                                     return_sequences=True))(dropout1)
    dropout2 = Dropout(0.5)(blstm2)

    dense1 = TimeDistributed(Dense(units=n_neurons_timedistributed,
                                   activation='relu'))(dropout2)
    dense2 = TimeDistributed(Dense(units=n_neurons_timedistributed,
                                   activation='relu'))(dense1)

    output = TimeDistributed(Dense(units=len(classes),
                                   activation='softmax'))(dense2)

    model = Model(inputs, output)
    print(model.summary())
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate))

    return model