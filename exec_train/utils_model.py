import transformers
transformers.logging.set_verbosity_error()

import os
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from urllib.parse import urlparse
from timm.models.hub import download_cached_file

# Set seed
def set_seed(seed=42):
    cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Is URL or not
def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

# Load checkpoint
def load_checkpoint(model, url_or_filename):
    # Check if URL or file
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('Checkpoint URL or Path is invalid!')

    # Load model
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Handle mismatch
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    # Display missing weights
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loading checkpoint from: {url_or_filename}")
    print("Missing keys in the loaded checkpoint:")
    for key in msg.missing_keys:
        print(f" - {key}")
    return model, checkpoint