import json
import pickle


def save_json(obj, file_path):
    with open(file_path, mode='w', encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def open_json(file_path):
    with open(file_path, mode='r', encoding="utf-8") as f:
        json_object = json.load(f)
        return json_object


def open_object(object_path):
    with open(object_path, mode='rb') as f:
        obj = pickle.load(f)

    return obj


def save_object(object_path, obj):
    with open(object_path, mode='wb') as f:
        pickle.dump(obj, f)
