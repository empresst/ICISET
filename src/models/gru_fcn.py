# src/models/gru_fcn.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dropout, Conv1D, BatchNormalization, Activation, GlobalMaxPooling1D, Dense, Permute
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def create_gru_fcn(input_shape, X_train, Y_train, X_test, Y_test, scaler):
    """Exact GRU-FCN code + added evaluation print"""
    
    def generate_GRU_FCN_model(input_shape, nb_class):
        inp = Input(shape=input_shape)
        x_r = GRU(8)(inp)
        x_r = Dropout(0.8)(x_r)
        
        y = Permute((2, 1))(inp)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalMaxPooling1D()(y)
        
        x = y
        output = Dense(nb_class, activation='linear')(x)
        model = Model(inp, output)
        model.summary()
        return model

    model5 = generate_GRU_FCN_model(input_shape, 1)
    model5.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    model5.fit(X_train, Y_train, epochs=20, batch_size=64,
               validation_split=0.2, verbose=1, shuffle=True)

    predictions5 = model5.predict(X_test)
    predictions5_inv = scaler.inverse_transform(predictions5)
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

    print('Test Mean Absolute Error (GRU-FCN):', mean_absolute_error(Y_test_inv, predictions5_inv))
    print('Test Root Mean Squared Error (GRU-FCN):', np.sqrt(mean_squared_error(Y_test_inv, predictions5_inv)))

    return predictions5_inv