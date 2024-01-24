import requests
import json
import os
from os.path import join as pjoin
from .utils import get_file_path, serialize_model_json, serialize_model_msgpack
# Base URL of the NestJS server
BASE_URL = 'http://localhost:3000/gateway'  # Adjust port if needed


def get_hello():
    response = requests.get(f'{BASE_URL}/hello')
    return response.text


def upload_file(file_path):
    files = {'file': open(file_path, 'rb')}
    response = requests.post(f'{BASE_URL}/upload', files=files)
    return response.json()


def init_ledger():
    response = requests.get(f'{BASE_URL}/initLedger')
    return response.json()


def get_model(model_id):
    response = requests.get(f'{BASE_URL}/model/{model_id}')
    model = response.json()['data']
    return model


def get_all_models():
    response = requests.get(f'{BASE_URL}/allModels')
    return response.json()


def create_model(model_data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(
        f'{BASE_URL}/model', headers=headers, data=json.dumps(model_data))
    return response.json()


def submit_model(model_id, model_params, first_n=0, last_n=0):
    # headers = {'Content-Type': 'application/octet-stream'}
    headers = {'Content-Type': 'text/plain'}

    # json_data = serialize_model_json(model_params, first_n, last_n)
    encoded_model = serialize_model_msgpack(model_params, first_n, last_n)
    response = requests.post(
        f'{BASE_URL}/local-model/{model_id}', headers=headers, data=encoded_model)
    return response.json()


def aggregate_models(model_id, round_number):
    response = requests.post(
        f'{BASE_URL}/aggregate/{model_id}', params={'round': round_number})
    return response.json()


# Example usage:
if __name__ == "__main__":
    # Call the get_hello endpoint
    hello_response = get_hello()
    print(hello_response)

    # Call the upload_file endpoint
    # file_path = get_file_path('./data/20MB_file.txt')
    # file_upload_response = upload_file(file_path)  # Update with the correct file path
    # print(file_upload_response)

    # # Initialize the ledger
    # init_ledger_response = init_ledger()
    # print(init_ledger_response)

    # # Get all models
    all_models_response = get_all_models()
    print(all_models_response)

    # # Create a new model
    # new_model = {
    #     'id': 'model123',
    #     'size': 5,
    #     'owner': 'Alice'
    # }
    # create_model_response = create_model(new_model)
    # print(create_model_response)
