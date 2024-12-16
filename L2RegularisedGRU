import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.regularizers import l1, l2  # Import both L1 and L2 for exploration
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
 
# Define hyperparameters
n_layers = 1  
n_units = 64  
learning_rate = 0.001  
dropout_rate = 0.2  
regularizer = l2(0.001)  
 
# Create the GRU model with hyperparameter flexibility
def create_gru_model(input_shape, n_layers=1, n_units=64, learning_rate=0.001,
                     dropout_rate=0, regularizer=None):
 
  model = tf.keras.Sequential()
 
  for _ in range(n_layers):
    return_sequences = True if n_layers > 1 else False  # Adjust for stacked layers
    model.add(GRU(n_units, activation='tanh', return_sequences=return_sequences, input_shape=input_shape))
    if dropout_rate > 0:
      model.add(Dropout(dropout_rate))  # Add dropout for regularization
 
  # Add regularization to Dense layer
  model.add(Dense(1, activation='linear', kernel_regularizer=regularizer))
 
  # Compile the model
  model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
 
  return model
 
# Create the model with appropriate input shape
model4 = create_gru_model(X_train3.shape[1:])
 
# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
 
# Train the model (adjust epochs and batch size based on dataset size and hardware)
history4 = model4.fit(X_train3, Y_train3, epochs=10, batch_size=256,  # May need more epochs
                       validation_data=(X_test3, Y_test3), callbacks=[early_stopping], verbose=1, shuffle=False)
 
# Print model summary
model4.summary()
 
# Make predictions (optional)
predictions = model4.predict(X_test3)
 
train_predict2 = model4.predict(X_train3)
test_predict2 = model4.predict(X_test3)
# invert predictions
train_predict2 = scaler.inverse_transform(train_predict2)
Y_train3 = scaler.inverse_transform([Y_train3])
test_predict2 = scaler.inverse_transform(test_predict2)
Y_test3 = scaler.inverse_transform([Y_test3])
 
print('Train Mean Absolute Error:', mean_absolute_error(Y_train3[0], train_predict2[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train3[0], train_predict2[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test3[0], test_predict2[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test3[0], test_predict2[:,0])))
 
rmse_rgru=np.sqrt(mean_squared_error(Y_test3[0], test_predict2[:,0]))
mae_rgru=mean_absolute_error(Y_test3[0], test_predict2[:,0])
