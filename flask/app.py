from flask import Flask
import pickle
import json

modelo = {}
app = Flask(__name__)

def load_model():
    with open('./models/finished_model_arima.model', "rb") as archivo_entrada:
        modelo_arima = pickle.load(archivo_entrada)	
    return modelo_arima

@app.route('/')
def hello_world():
	return 'Endpoint Pickle loaded!\n\n'

@app.route('/predict')
def predict():
    predictions = modelo.predict(10)
    result = dict(enumerate(predictions.flatten(), 1))
    app_json = json.dumps(result)
    return app_json

if __name__ == "__main__":
    modelo = load_model()
    app.run()

