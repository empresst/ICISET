# src/models/cnn_gru.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, GRU, Dense, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def create_cnn_gru(input_shape, X_train, Y_train, X_test, Y_test, scaler):
    """Exact CNN-GRU code + added evaluation print"""
    def generate_CNN_GRU_model(input_shape):
        inp = Input(shape=input_shape)
        x = Conv1D(64, 5, activation='relu')(inp)
        x = MaxPooling1D(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = GRU(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=output)
        model.summary()
        return model

    model_cnn_gru = generate_CNN_GRU_model(input_shape)
    model_cnn_gru.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    model_cnn_gru.fit(X_train_reshaped, Y_train, epochs=20, batch_size=256,
                      validation_split=0.2, verbose=1, shuffle=True)

    test_predict = model_cnn_gru.predict(X_test_reshaped)
    test_predict_inv = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

    print('Test Mean Absolute Error (CNN-GRU):', mean_absolute_error(Y_test_inv, test_predict_inv))
    print('Test Root Mean Squared Error (CNN-GRU):', np.sqrt(mean_squared_error(Y_test_inv, test_predict_inv)))

    return test_predict_inv