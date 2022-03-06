from flask import Flask, render_template, request
import pickle
import json
import boto3
import logging

app = Flask(__name__)
gunicorn_error_logger = logging.getLogger('gunicorn.error')
app.logger.handlers.extend(gunicorn_error_logger.handlers)
app.logger.setLevel(logging.INFO)

def load_model(aws_info):
    # Creating the low level functional client
# Creating the high level object oriented interface
    app.logger.info('Loading file ' + aws_info['model_file'] + ' from S3')
    resource = boto3.resource(
        's3',
        aws_access_key_id = aws_info["key"] ,
        aws_secret_access_key = aws_info["secret"] ,
        region_name = aws_info['aws_region']
    )

    with open('local_model.pkl', 'wb') as data:
        resource.Bucket(aws_info['s3_bucket']).download_fileobj(aws_info['model_file'], data)  

    app.logger.info('Unpickling file ' + aws_info['model_file'] + ' to local_model.pkl')
    with open('local_model.pkl', 'rb') as data:
        modelo_arima = pickle.load(data)

    app.logger.info('Model ready')
    return modelo_arima

def get_aws_creds():
    myvars = {}
    with open("aws_info.txt") as myfile:
        for line in myfile:
            name, var = line.partition("=")[::2]
            myvars[name.strip()] = var.rstrip()

    return myvars

modelo = load_model(get_aws_creds())

@app.route('/')
def hello_world():
	return 'Endpoint Pickle loaded!\n\n'


@app.route('/predict/<int:periods>')
def predict(periods):
    predictions = modelo.predict(periods)
    result = dict(enumerate(predictions.flatten(), 1))
    app_json = json.dumps(result)
    return app_json

if __name__ == "__main__":
    app.run()

