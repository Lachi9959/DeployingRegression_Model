# === 1. Data Preprocessing and Model Training ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load Dataset (you can replace this with your own CSV file)
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

# Features and Target
X = df.drop('medv', axis=1)  # 'medv' is the price
y = df['medv']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)

# Save the model
joblib.dump(model, 'model.pkl')


# === 2. Flask App Code (app.py) ===
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('model.pkl')

# Extract feature names from training data
feature_names = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
                 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to House Price Prediction API! Use /predict to POST data."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        features = [data[col] for col in feature_names]
        prediction = model.predict([features])
        return jsonify({"Predicted Price": round(prediction[0], 2)})
    except KeyError as e:
        return jsonify({"error": f"Missing input feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# === 3. Sample JSON Payload for Testing ===
# POST to http://localhost:5000/predict
# {
#     "crim": 0.1,
#     "zn": 18.0,
#     "indus": 2.31,
#     "chas": 0,
#     "nox": 0.538,
#     "rm": 6.575,
#     "age": 65.2,
#     "dis": 4.09,
#     "rad": 1,
#     "tax": 296.0,
#     "ptratio": 15.3,
#     "b": 396.9,
#     "lstat": 4.98
# }