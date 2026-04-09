model1 = Sequential()

# Add the Bi-LSTM layer with 100 units in each direction
model1.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(X_train2.shape[1], X_train2.shape[2])))

# Add dropout for regularization
model1.add(Dropout(0.2))

# Add another Bi-LSTM layer, no need for return_sequences here as it's the last LSTM layer
model1.add(Bidirectional(LSTM(100)))

# Add the output layer
model1.add(Dense(1))

# Compile the model
model1.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
history2 = model1.fit(X_train2, Y_train2, epochs=10, batch_size=1600, validation_data=(X_test2, Y_test2),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=4)], verbose=1, shuffle=False)

# Display model summary
model1.summary()

train_predict1 = model1.predict(X_train2)
test_predict1 = model1.predict(X_test2)
# invert predictions
train_predict1 = scaler.inverse_transform(train_predict1)
Y_train2 = scaler.inverse_transform([Y_train2])
test_predict1 = scaler.inverse_transform(test_predict1)
Y_test2 = scaler.inverse_transform([Y_test2])

print('Train Mean Absolute Error:', mean_absolute_error(Y_train2[0], train_predict1[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train2[0], train_predict1[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test2[0], test_predict1[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test2[0], test_predict1[:,0])))
