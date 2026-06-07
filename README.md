# weather_app
🌤️ AI Weather Prediction & Forecasting System
Show Image
Show Image
Show Image
Show Image

Dual-model ML system for temperature regression and weather category classification, deployed via Flask.


📌 Overview
A machine learning-powered weather forecasting system that predicts both exact temperature values (regression) and weather categories (classification) from environmental input features. Built with dual Random Forest models and deployed as a lightweight Flask web application with confidence scoring, making weather intelligence accessible through a clean web interface.

✨ Features

🌡️ Temperature Prediction — RF Regressor forecasts exact temperature values
🌦️ Weather Classification — RF Classifier categorizes weather (Sunny / Cloudy / Rainy / Stormy)
🎯 Confidence Scoring — Each prediction includes a probability confidence score
🌐 Flask Web Deployment — Accessible via browser with REST-style form submission
💾 Pickle Serialization — Lightweight, fast model loading with no heavy dependencies
📈 Feature-Rich Input — Accepts humidity, pressure, wind speed, dew point, and more


🏗️ Architecture
User Input (Weather Parameters)
          │
          ▼
   Flask Web Server
          │
    ┌─────┴─────┐
    ▼           ▼
 RF Regressor  RF Classifier
 (Temperature) (Weather Type)
    │           │
    ▼           ▼
  Temp Value  Category + Confidence
    │           │
    └─────┬─────┘
          ▼
   Results Page (Flask Template)

🧰 Tech Stack
LayerTechnologyBackendFlaskML ModelsRandom Forest Regressor + ClassifierModel StoragePickleData ProcessingPandas, NumPyFrontendHTML5, CSS3 (Jinja2 Templates)LanguagePython 3.9+

🚀 Getting Started
Prerequisites
bashPython 3.9+
pip
Installation
bash# Clone the repository
git clone https://github.com/Priyankasivalingam/ai-weather-prediction.git
cd ai-weather-prediction

# Install dependencies
pip install -r requirements.txt

# Train models (if not pre-trained)
python train_models.py

# Run Flask server
python app.py
Requirements
flask
scikit-learn
pandas
numpy
pickle5


🌐 API Endpoints
EndpointMethodDescription/GETHome page with input form/predictPOSTRun dual prediction (temp + type)/api/predictPOSTJSON API for programmatic access
Sample API Request
jsonPOST /api/predict
{
  "humidity": 75,
  "pressure": 1013,
  "wind_speed": 12,
  "dew_point": 18,
  "cloud_cover": 60
}
Sample API Response
json{
  "temperature": 24.3,
  "weather_type": "Cloudy",
  "confidence": 0.87
}

📊 Model Performance
ModelMetricScoreRegressorR² Score~0.91RegressorMAE~1.4°CClassifierAccuracy~92%ClassifierF1-Score~91%

🔮 Future Improvements

 Integration with live weather APIs (OpenWeatherMap)
 7-day multi-step forecasting
 Location-based predictions (GPS input)
 Time-series LSTM model comparison


👩‍💻 Author
Priyanka S — LinkedIn · GitHub

📄 License
This project is licensed under the MIT License.
