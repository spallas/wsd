import tensorflow as tf

from keras import Model, Input
from keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional
from sklearn.metrics import confusion_matrix, classification_report

from data_preprocessing import load_dataset, load_embeddings, get_embeddings


def train_and_test(vocab_size, embed_size, lstm_size):

    x_train, y_train, x_devel, y_devel = load_dataset()
    embeddings = get_embeddings()

    senses_size = 10000  # # # #
    max_sentence_len = 30  # # # #
    input_size = max_sentence_len

    embedding_layer = Embedding(vocab_size,
                                embed_size,
                                weights=[embeddings],
                                input_length=input_size,
                                trainable=False)

    input_layer = Input(shape=(input_size,), dtype='float32')
    embeddings_output = embedding_layer(input_layer)

    x = Bidirectional(LSTM(lstm_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(embeddings_output)
    pred_layer = Dense(256, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=pred_layer)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=16,
              validation_data=(x_devel, y_devel))

    model.save("wsd_model_keras.h5")
