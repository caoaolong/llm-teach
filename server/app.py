from flask import Flask
from flask_cors import CORS
from apps.mnist import load_model_endpoint as mnist_load_model, predict as mnist_predict
from apps.sentiment import load_model_endpoint as sentiment_load_model, predict as sentiment_predict

app = Flask(__name__)
CORS(app)

@app.route('/mnist/predict', methods=['POST'])
def mnist_predict_route():
    return mnist_predict()

@app.route('/mnist/load', methods=['POST'])
def mnist_load_model_route():
    return mnist_load_model()

@app.route('/sentiment/predict', methods=['POST'])
def sentiment_predict_route():
    return sentiment_predict()

@app.route('/sentiment/load', methods=['POST'])
def sentiment_load_model_route():
    return sentiment_load_model()

if __name__ == '__main__':
    app.run(port=5000, debug=True)