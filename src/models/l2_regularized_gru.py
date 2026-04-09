# src/models/l2_regularized_gru.py
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def create_l2_regularized_gru(input_shape, X_train, Y_train, X_test, Y_test, scaler):
    """Exact L2 Regularised GRU code from your notebook"""
    def create_gru_model(input_shape):
        model = tf.keras.Sequential()
        model.add(GRU(64, activation='tanh', input_shape=input_shape, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear', kernel_regularizer=l2(0.001)))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
        return model

    model4 = create_gru_model(input_shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)

    model4.fit(X_train, Y_train, epochs=10, batch_size=256,
               validation_data=(X_test, Y_test),
               callbacks=[early_stopping], verbose=1, shuffle=False)

    test_predict2 = model4.predict(X_test)
    test_predict2 = scaler.inverse_transform(test_predict2)
    return test_predict2