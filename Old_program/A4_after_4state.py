import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# データの読み込み
data = pd.read_csv('Book2023_CAL.csv')

# 特徴量の抽出と標準化
X = data[["year","GDP","poplation(20-)(male)","poplation(20+)(male)","poplation(65+)(male)","poplation(20-)(female)","poplation(20+)(female)","poplation(65+)(female)","employment(20-)(female)","employment(20+)(female)","employment(65+)(female)","employment(20-)(male)","employment(20+)(male)","employment(65+)(male)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 学習済みモデルのロード
loaded_model = tf.keras.models.load_model('../tensorflow/asset')

OUTPUT_file = open("../PREDICT_OUTPUT.csv", "w")

# 新しいデータの入力
while True:
    new_data = []
    user_input = input("Enter a comma-separated list of numbers: ")

    # 改行文字を削除してから処理を実行
    user_input = user_input.replace('\n', '')
    #line_string = input("Paste here:")
    new_data = [float(number_data) for number_data in user_input.split(",")]
    #for i in range(X.shape[1]):
    #    value = float(input(str(i) + ":"))
    #    if value == 999:
    #        break
    #    new_data.append(value)

    #if value == 999:
    #    break

    for Current_data in new_data:
        OUTPUT_file.write(str(Current_data) + ",")

    new_data = np.array([new_data])
    new_data_scaled = scaler.transform(new_data)
    prediction = loaded_model.predict(new_data_scaled)
    print(f'Predicted Crime for the new data: {prediction[0][0]}')

    OUTPUT_file.write(str(prediction[0][0])+"\n")