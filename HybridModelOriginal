import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, LSTM, GRU, Bidirectional, Input, Conv1D, GlobalMaxPooling1D, LeakyReLU
from tensorflow.keras.models import Model
from adabelief_tf import AdaBeliefOptimizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tcn import TCN

# Define hyperparameters
lstm_units = 64
gru_units = 64
tcn_filters = 64
tcn_kernel_size = 3
initial_learning_rate = 0.001
dropout_rate = 0.1
leaky_relu_alpha = 0.01  # Alpha value for LeakyReLU

# Define the LSTM model
def create_lstm_model(input_shape, units=lstm_units):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(units, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(units))(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.01)(x)
    model = Model(inputs=inputs, outputs=x)
    optimizer = AdaBeliefOptimizer(learning_rate=initial_learning_rate, epsilon=1e-14, rectify=True)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Define the GRU model
def create_gru_model(input_shape, units=gru_units):
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(units, return_sequences=True))(inputs)
    x = Bidirectional(GRU(units))(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.01)(x)
    model = Model(inputs=inputs, outputs=x)
    optimizer = AdaBeliefOptimizer(learning_rate=initial_learning_rate, epsilon=1e-14, rectify=True)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Define the TCN model
def create_tcn_model(input_shape, filters=tcn_filters, kernel_size=tcn_kernel_size):
    inputs = Input(shape=input_shape)
    x = inputs
    for dilation_rate in [4, 8]:
        x = Conv1D(filters, kernel_size, activation='relu', padding='causal', dilation_rate=dilation_rate)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.01)(x)
    model = Model(inputs=inputs, outputs=x)
    optimizer = AdaBeliefOptimizer(learning_rate=initial_learning_rate, epsilon=1e-14, rectify=True)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Create instances of the LSTM, GRU, and TCN models
input_shape_lstm = (X_train4.shape[1], X_train4.shape[2])
input_shape_gru = (X_train4.shape[1], X_train4.shape[2])
input_shape_tcn = (X_train4.shape[1], X_train4.shape[2])

lstm_model = create_lstm_model(input_shape_lstm)
gru_model = create_gru_model(input_shape_gru)
tcn_model_instance = create_tcn_model(input_shape_tcn)

# Define the combined model
def combine_models(lstm_model, gru_model, tcn_model):
    combined_output = Concatenate()([lstm_model.output, gru_model.output, tcn_model.output])
    combined_output = Dense(256)(combined_output)
    combined_output = LeakyReLU(alpha=0.01)(combined_output)
    combined_output = Dense(128)(combined_output)
    combined_output = LeakyReLU(alpha=0.01)(combined_output)
    combined_output = Dense(64)(combined_output)
    combined_output = LeakyReLU(alpha=0.01)(combined_output)
    combined_output = Dense(32)(combined_output)
    combined_output = LeakyReLU(alpha=0.01)(combined_output)
    combined_output = Dense(1)(combined_output)
    combined_model = Model(inputs=[lstm_model.input, gru_model.input, tcn_model.input], outputs=combined_output)
    return combined_model

combined_model = combine_models(lstm_model, gru_model, tcn_model_instance)

# Define learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

# Compile the combined model with AdaBelief optimizer
combined_model.compile(loss='mean_squared_error', optimizer=AdaBeliefOptimizer(learning_rate=initial_learning_rate, epsilon=1e-14, rectify=True))

# Train the combined model with learning rate scheduler
history_combined = combined_model.fit([X_train4, X_train4, X_train4], Y_train4, epochs=20, batch_size=256,
                                      validation_data=([X_test4, X_test4, X_test4], Y_test4),
                                      callbacks=[EarlyStopping(monitor='val_loss', patience=4),
                                                 reduce_lr],
                                      verbose=1, shuffle=True)

# Display model summary
combined_model.summary()

# Predictions
lstm_predictions_test = lstm_model.predict(X_test4)
gru_predictions_test = gru_model.predict(X_test4)
tcn_predictions_test = tcn_model_instance.predict(X_test4)
comb_test = combined_model.predict([X_test4, X_test4, X_test4])

lstm_predictions_train = lstm_model.predict(X_train4)
gru_predictions_train = gru_model.predict(X_train4)
tcn_predictions_train = tcn_model_instance.predict(X_train4)
comb_train = combined_model.predict([X_train4, X_train4, X_train4])



# Combine predictions with original test features
combined_features_test = np.concatenate([X_test4.reshape(X_test4.shape[0], -1), comb_test, lstm_predictions_test, gru_predictions_test, tcn_predictions_test], axis=1)
combined_features_train = np.concatenate([X_train4.reshape(X_train4.shape[0], -1), comb_train, lstm_predictions_train, gru_predictions_train, tcn_predictions_train], axis=1)
from xgboost import XGBRegressor

# Define XGBRegressor model
xgb_model = XGBRegressor()

# Train XGBRegressor on combined features
xgb_model.fit(combined_features_train, Y_train4)

test_predictions = xgb_model.predict(combined_features_test)

# Inverse transform the predictions and actual values
test_predictions_inv = scaler.inverse_transform(test_predictions.reshape(-1, 1))  # Reshape predictions to match scaler dimensions
Y_test4_inv = scaler.inverse_transform(Y_test4.reshape(-1, 1))  # Reshape actual values to match scaler dimensions

# Calculate evaluation metrics
test_mae = mean_absolute_error(Y_test4_inv, test_predictions_inv)
test_rmse = np.sqrt(mean_squared_error(Y_test4_inv, test_predictions_inv))


print('Test Mean Absolute Error:', test_mae)
print('Test Root Mean Squared Error:', test_rmse)

comb_test_inv = scaler.inverse_transform(comb_test);

tm = mean_absolute_error( Y_test4_inv, comb_test_inv)
print(tm)
tr = np.sqrt(mean_squared_error( Y_test4_inv, comb_test_inv))
print(tr)
