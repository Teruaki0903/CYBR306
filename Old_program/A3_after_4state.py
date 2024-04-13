import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Book2023_CAL.csv')

X = data[["year","GDP","poplation(20-)(male)","poplation(20+)(male)","poplation(65+)(male)","poplation(20-)(female)","poplation(20+)(female)","poplation(65+)(female)","employment(20-)(female)","employment(20+)(female)","employment(65+)(female)","employment(20-)(male)","employment(20+)(male)","employment(65+)(male)"]]
y = data['crime']

feature_names = X.columns

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
while True:
    list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(14):
        list[i] = input(str(i)+":")
        if list[i] == "999":
            model.save('tensorflow/asset', save_format='tf')
            break
    if list[0] == "999":
        break

    new_data = np.array([list])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    print(f'Predicted Crime for the new data: {prediction[0][0]}')
