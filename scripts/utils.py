import json

def load_config():
    try:
        with open('scripts/config.json', 'r') as f:
            return json.load(f)
    except:
        print("No config file")
