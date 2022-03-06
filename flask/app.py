from flask import Flask
import pickle

app = Flask(__name__)

@app.route('/')
def hello_world():
	return 'Hello World Pickle loaded!\n\n'

if __name__ == "__main__":
    with open('./models/finished_model_arima.model', "rb") as archivo_entrada:
        modelo_arima = pickle.load(archivo_entrada)	
    app.run()

