from pathlib import Path

# Get root directory
def get_root() -> Path:
    return Path(__file__).resolve().parent.parent

# Get configs directory
def get_configs():
    return get_root() / 'configs'

# Get data directory
def get_data():
    return get_root() / 'data'

# Get store model directory
def get_store_model():
    return get_root() / 'store_model'