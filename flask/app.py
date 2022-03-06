from flask import Flask, render_template, request, jsonify
import pickle
import json
import boto3
import logging
import plotly.express as px
from plotly.io import to_html
from datetime import datetime, timedelta


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

def make_predictions2(modelo_arima, n_periods=2):
    prediccion_model2 = list(modelo_arima.predict(n_periods=n_periods))
    # print(prediccion_model2)
    return prediccion_model2

def get_dates(periods):
    last_date = datetime.strptime('2019-12-31 20:00', '%Y-%m-%d %H:%M')
    min_diff = 240  # 4 hours
    dates = []

    for i in range(1, periods + 1):
        time_change = timedelta(minutes=min_diff * i)
        new_date = last_date + time_change
        dates.append(new_date.strftime('%Y-%m-%d %H:%M'))

    return dates

def make_graph(predictions, periods):
    x = get_dates(periods)
    y = predictions

    fig = px.line(x=x, y=y, title='Particles evolution')

    return to_html(fig, include_plotlyjs=False, include_mathjax=False, full_html=False)

app.logger.info('App root path: ' + app.root_path)
modelo = load_model(get_aws_creds())

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict/<int:periods>')
def predict(periods):
    predictions = modelo.predict(periods)
    result = dict(enumerate(predictions.flatten(), 1))
    app_json = json.dumps(result)
    return app_json

@app.route('/api/model/unico/<int:periods>', methods=['GET'])
def predict2(periods):
    prediccion_model2 = make_predictions2(modelo, int(periods))

    response = {
        'prediction': prediccion_model2,
        'graph': make_graph(prediccion_model2, periods)
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run()

