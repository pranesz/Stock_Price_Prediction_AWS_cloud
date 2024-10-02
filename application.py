from flask import Flask, render_template, request, jsonify 
import joblib
import numpy as np


model = joblib.load('model/Stock_price_prediction.pkl')

application = Flask(__name__)

@application.route('/')
def Home():
    return render_template('Home.html')

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([[data['prev_close_1'], data['prev_close_2'], data['prev_close_3']]])
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})
if __name__ == '__main__':
    application.run(debug=True)
