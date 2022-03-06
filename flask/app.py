from flask import Flask, render_template, request
import pickle
import json
import boto3

app = Flask(__name__)

def load_model(creds):
    # Creating the low level functional client
# Creating the high level object oriented interface
    resource = boto3.resource(
        's3',
        aws_access_key_id = creds["key"] ,
        aws_secret_access_key = creds["secret"] ,
        region_name = 'eu-west-3'
    )

    with open('local_model.pkl', 'wb') as data:
        resource.Bucket("dsftnov21-prod-test03").download_fileobj("finished_model_arima.model", data)  

    with open('local_model.pkl', 'rb') as data:
        modelo_arima = pickle.load(data)

    """with open('./models/finished_model_arima.model', "rb") as archivo_entrada:
        modelo_arima = pickle.load(archivo_entrada)	"""

    return modelo_arima

def get_aws_creds():
    myvars = {}
    with open("aws_creds.txt") as myfile:
        for line in myfile:
            name, var = line.partition("=")[::2]
            myvars[name.strip()] = var.rstrip()

    return myvars

modelo = load_model(get_aws_creds())

@app.route('/')
def hello_world():
	return 'Endpoint Pickle loaded!\n\n'

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict')
def predict():
    predictions = modelo.predict(10)
    result = dict(enumerate(predictions.flatten(), 1))
    app_json = json.dumps(result)
    return app_json

if __name__ == "__main__":
    app.run()

