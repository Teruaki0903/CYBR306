import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import SLR

# Read data
data = pd.read_csv('Book2033_ALLSTATE.csv')

slr = SLR.SimpleLinearRegression()

# Get data label
X = data[["GDP","poplation(20-)(male)","poplation(20+)(male)","poplation(65+)(male)","poplation(20-)(female)","poplation(20+)(female)","poplation(65+)(female)","employment(20-)(female)","employment(20+)(female)","employment(65+)(female)","employment(20-)(male)","employment(20+)(male)","employment(65+)(male)"]]
# #Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Load saved model
loaded_model = tf.keras.models.load_model('tensorflow/asset')
#open CSV file (for Save outpu)
OUTPUT_file = open("PREDICT_OUTPUT.csv", "a")
# Predict part
while True:
    # new_data = []
    user_input_State = str(input("Enter two upper letter of state abbreviation to predict (To quit, enter '999'): "))
    user_input_year = int(input("Enter a year to predict: "))
    if user_input_State == "999":
        break
    new_data = slr.submit_prediction(user_input_State, user_input_year)
    for Current_data in new_data:
        OUTPUT_file.write(str(Current_data) + ",")
    #input data to AI model
    new_data = np.array([new_data])
    new_data_scaled = scaler.transform(new_data)
    prediction = loaded_model.predict(new_data_scaled)
    print(f'Predicted Crime for the new data: {prediction[0][0]}')
    #append Predit
    OUTPUT_file.write(str(prediction[0][0])+"\n")