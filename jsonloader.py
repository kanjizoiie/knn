import json
import io

def load_file(filename):
    file = open(filename, "r")
    if not(file.closed):
        text = file.read()
        file.close()
        json_data = json.loads(text)
        return json_data
    else:
        raise Exception("The file was not opened")


def save_file(filename, json_data):
    text = json.dumps(json_data)
    file = open(filename, "w")
    file.write(text)
    file.close()