import json
import os


def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
