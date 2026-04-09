# src/models/hybrid_ensemble.py
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, LSTM, GRU, Bidirectional, Input, Conv1D, GlobalMaxPooling1D, LeakyReLU
from tensorflow.keras.models import Model
from adabelief_tf import AdaBeliefOptimizer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_hybrid(X_train, Y_train, X_test, Y_test, scaler):
    """Exact Hybrid (BiLSTM + BiGRU + TCN + XGBoost) code from your notebook"""
    
    lstm_units = 64
    gru_units = 64
    tcn_filters = 64
    tcn_kernel_size = 3
    initial_learning_rate = 0.001
    leaky_relu_alpha = 0.01

    def create_lstm_model(input_shape, units=lstm_units):
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(units, return_sequences=True))(inputs)
        x = Bidirectional(LSTM(units))(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=leaky_relu_alpha)(x)
        model = Model(inputs=inputs, outputs=x)
        optimizer = AdaBeliefOptimizer(learning_rate=initial_learning_rate, epsilon=1e-14, rectify=True)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def create_gru_model(input_shape, units=gru_units):
        inputs = Input(shape=input_shape)
        x = Bidirectional(GRU(units, return_sequences=True))(inputs)
        x = Bidirectional(GRU(units))(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=leaky_relu_alpha)(x)
        model = Model(inputs=inputs, outputs=x)
        optimizer = AdaBeliefOptimizer(learning_rate=initial_learning_rate, epsilon=1e-14, rectify=True)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def create_tcn_model(input_shape, filters=tcn_filters, kernel_size=tcn_kernel_size):
        inputs = Input(shape=input_shape)
        x = inputs
        for dilation_rate in [4, 8]:
            x = Conv1D(filters, kernel_size, activation='relu', padding='causal', dilation_rate=dilation_rate)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=leaky_relu_alpha)(x)
        model = Model(inputs=inputs, outputs=x)
        optimizer = AdaBeliefOptimizer(learning_rate=initial_learning_rate, epsilon=1e-14, rectify=True)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = create_lstm_model(input_shape)
    gru_model = create_gru_model(input_shape)
    tcn_model_instance = create_tcn_model(input_shape)

    def combine_models(lstm_model, gru_model, tcn_model):
        combined_output = Concatenate()([lstm_model.output, gru_model.output, tcn_model.output])
        combined_output = Dense(256)(combined_output)
        combined_output = LeakyReLU(alpha=leaky_relu_alpha)(combined_output)
        combined_output = Dense(128)(combined_output)
        combined_output = LeakyReLU(alpha=leaky_relu_alpha)(combined_output)
        combined_output = Dense(64)(combined_output)
        combined_output = LeakyReLU(alpha=leaky_relu_alpha)(combined_output)
        combined_output = Dense(32)(combined_output)
        combined_output = LeakyReLU(alpha=leaky_relu_alpha)(combined_output)
        combined_output = Dense(1)(combined_output)
        combined_model = Model(inputs=[lstm_model.input, gru_model.input, tcn_model.input], outputs=combined_output)
        return combined_model

    combined_model = combine_models(lstm_model, gru_model, tcn_model_instance)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    combined_model.compile(loss='mean_squared_error', 
                           optimizer=AdaBeliefOptimizer(learning_rate=initial_learning_rate, epsilon=1e-14, rectify=True))

    combined_model.fit([X_train, X_train, X_train], Y_train, epochs=20, batch_size=256,
                       validation_data=([X_test, X_test, X_test], Y_test),
                       callbacks=[EarlyStopping(monitor='val_loss', patience=4), reduce_lr],
                       verbose=1, shuffle=True)

    combined_model.summary()

    lstm_predictions_test = lstm_model.predict(X_test)
    gru_predictions_test = gru_model.predict(X_test)
    tcn_predictions_test = tcn_model_instance.predict(X_test)
    comb_test = combined_model.predict([X_test, X_test, X_test])

    lstm_predictions_train = lstm_model.predict(X_train)
    gru_predictions_train = gru_model.predict(X_train)
    tcn_predictions_train = tcn_model_instance.predict(X_train)
    comb_train = combined_model.predict([X_train, X_train, X_train])

    combined_features_test = np.concatenate([X_test.reshape(X_test.shape[0], -1),
                                             comb_test, lstm_predictions_test,
                                             gru_predictions_test, tcn_predictions_test], axis=1)
    combined_features_train = np.concatenate([X_train.reshape(X_train.shape[0], -1),
                                              comb_train, lstm_predictions_train,
                                              gru_predictions_train, tcn_predictions_train], axis=1)

    xgb_model = XGBRegressor()
    xgb_model.fit(combined_features_train, Y_train)

    test_predictions = xgb_model.predict(combined_features_test)

    test_predictions_inv = scaler.inverse_transform(test_predictions.reshape(-1, 1))
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

    print('Test Mean Absolute Error (Hybrid):', mean_absolute_error(Y_test_inv, test_predictions_inv))
    print('Test Root Mean Squared Error (Hybrid):', np.sqrt(mean_squared_error(Y_test_inv, test_predictions_inv)))

    return test_predictions_inv