import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#read data
data = pd.read_csv('Book2033_ALLSTATE.csv')
#get data label
X = data[["GDP","poplation(20-)(male)","poplation(20+)(male)","poplation(65+)(male)","poplation(20-)(female)","poplation(20+)(female)","poplation(65+)(female)","employment(20-)(female)","employment(20+)(female)","employment(65+)(female)","employment(20-)(male)","employment(20+)(male)","employment(65+)(male)"]]
y = data['crime']
#------------------------------------------------------------------
#Data preprocessing
feature_names = X.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
#------------------------------------------------------------------
#Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
#Build AI model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1)
])
#Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')
#calculate loss
model.fit(X_train, y_train, epochs=300000, batch_size=16, validation_data=(X_test, y_test))
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
while True:
    new_data = []
    user_input = input("Enter a comma-separated list of numbers: ")
    #save model
    if user_input == "999":
        model.save('tensorflow/asset', save_format='tf')
        break

    # 改行文字を削除してから処理を実行
    user_input = user_input.replace('\n', '')
    # line_string = input("Paste here:")
    new_data = [float(number_data) for number_data in user_input.split(",")]

    # Test part
    new_data = np.array([new_data])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    print(f'Predicted Crime for the new data: {prediction[0][0]}')