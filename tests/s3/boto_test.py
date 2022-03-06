import boto3
import pandas
import pickle

"""
https://supsystic.com/documentation/id-secret-access-key-amazon-s3/
https://www.sqlshack.com/getting-started-with-amazon-s3-and-python/
https://stackoverflow.com/questions/48964181/how-to-load-a-pickle-file-from-s3-to-use-in-aws-lambda
https://www.atmosera.com/blog/creating-machine-learning-web-api-flask/
https://github.com/Wintellect/DataScienceExamples/blob/master/Regression/SimpleLinearRegressionAPI.py

"""



# Creating the low level functional client
client = boto3.client(
    's3',
    aws_access_key_id = 'AKIAQJD4VX47LN47IVP7',
    aws_secret_access_key = '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',
    region_name = 'eu-west-3'
)
    
# Creating the high level object oriented interface
resource = boto3.resource(
    's3',
    aws_access_key_id = 'AKIAQJD4VX47LN47IVP7',
    aws_secret_access_key = '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',
    region_name = 'eu-west-3'
)


# Fetch the list of existing buckets
clientResponse = client.list_buckets()
    
# Print the bucket names one by one
print('Printing bucket names...')
for bucket in clientResponse['Buckets']:
    print(f'Bucket Name: {bucket["Name"]}')


# Creating a bucket in AWS S3
"""location = {'LocationConstraint': 'eu-west-3'}
client.create_bucket(
    Bucket='dsftnov21-prod-test03',
    CreateBucketConfiguration=location
)"""


obj = client.get_object(
    Bucket = 'dsftnov21-prod-test03',
    Key = 'finished_model_arima.model'
)

with open('local_model.pkl', 'wb') as data:
    resource.Bucket("dsftnov21-prod-test03").download_fileobj("finished_model_arima.model", data)  

with open('local_model.pkl', 'rb') as data:
    old_list = pickle.load(data)