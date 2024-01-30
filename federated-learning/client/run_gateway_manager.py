# !/usr/bin/env python
import argparse
import sys
from .services.gateway_client import delete_all_models, get_all_model_names, delete_model


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--delete_all', action='store_true',
                        help='Deletes all the models stored in the blockchain.')
    parser.add_argument('--delete', type=str, default=None,
                        help='Deletes the model stored in the blockchain with the corresponding id.')
    parser.add_argument('--get_all', action='store_true',
                        help='Retrieves all model names stored in the blockchain.')
    
    args = parser.parse_args()

    if args.delete_all:
        delete_all()
        sys.exit(0)

    if args.get_all:
        get_all_models()
        sys.exit(0)

    return args


def delete_all():
    response = delete_all_models()
    print(response)


def delete(model_id):
    response = delete_model(model_id)
    print(response)


def get_all_models():
    response = get_all_model_names()
    print(response)


def main():
    args = parse_args()

    if args.delete is not None:
        delete(args.delete)
        sys.exit(0)


if __name__ == "__main__":
    main()
