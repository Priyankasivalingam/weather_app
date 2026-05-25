from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load models
temp_model = pickle.load(open("model/temp_model.pkl", "rb"))
weather_model = pickle.load(open("model/weather_model.pkl", "rb"))
le = pickle.load(open("model/label_encoder.pkl", "rb"))

with open("model/rmse.txt", "r") as f:
    rmse = f.read()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temp'])
        dew = float(request.form['dew'])
        humidity = float(request.form['humidity'])
        wind = float(request.form['wind'])
        visibility = float(request.form['visibility'])
        pressure = float(request.form['pressure'])

        # Feature Engineering
        temp_diff = temp - dew
        wind_vis = wind / (visibility + 1)
        pressure_norm = pressure / 100

        input_data = np.array([[temp, dew, humidity, wind, visibility,
                                pressure, temp_diff, wind_vis, pressure_norm]])

        # Predictions
        temp_pred = temp_model.predict(input_data)[0]

        weather_pred = weather_model.predict(input_data)[0]
        weather_text = le.inverse_transform([weather_pred])[0]

        # Confidence (Advanced)
        tree_preds = [tree.predict(input_data)[0] for tree in temp_model.estimators_]
        confidence = np.std(tree_preds)

        result = {
            "temperature": round(temp_pred, 2),
            "weather": weather_text,
            "rmse": rmse,
            "confidence": round(confidence, 2)
        }

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)