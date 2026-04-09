from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Input, Embedding, GRU, LSTM, MaxPooling1D, GlobalMaxPool1D
from keras.layers import Dropout, Dense, Activation, Flatten,Conv1D, Bidirectional, SpatialDropout1D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
 
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
 
# Assuming X_train.shape = (num_samples, num_timesteps, num_features)
input_shape = (30, 1)  # Adjust the input shape to match the data shape
 
# Generate the CNN-GRU model
model_cnn_gru = generate_CNN_GRU_model(input_shape)
 
# Compile the model
model_cnn_gru.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])
 
# Reshape the input data to match the model input shape
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
 
# Train the model
history_cnn_gru = model_cnn_gru.fit(X_train_reshaped, Y_train, epochs=20, batch_size=256, validation_split=0.2, verbose=1, shuffle=True)
 
# Predict on test set
 
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])
 
# Make predictions
train_predict = model_cnn_gru.predict(X_train_reshaped)
test_predict = model_cnn_gru.predict(X_test_reshaped)
 
# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))
 
# Calculate and print errors
print('Train Mean Absolute Error:', mean_absolute_error(Y_train, train_predict))
print('Train Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_train, train_predict)))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test, test_predict))
mae_cnngru = mean_absolute_error(Y_test, test_predict)
print('Test Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, test_predict)))
rmse_cnngru = np.sqrt(mean_squared_error(Y_test, test_predict))
