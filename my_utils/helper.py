import os
import pickle
import json
from huggingface_hub import hf_hub_download
from transformers import Data2VecAudioConfig
from my_utils.config import ENCODERS_CONFIG

def get_model_dir(data_config, model_config):
    ds_name = data_config["ds_name"]
    encoder_name = model_config["encoder_name"]
    encoder_size = model_config["encoder_size"]
    freeze_encoder = model_config["freeze_encoder"]
    freeze_feature_extractor = model_config["freeze_feature_extractor"]
    lora_encoder = model_config["lora_encoder"]
    layer_selector_encoder = model_config["layer_selection_encoder"]

    if encoder_size == "default": 
        encoder_size = ENCODERS_CONFIG[encoder_name]["default_size"] 

    model_dir = f"weights/{ds_name}_{encoder_name}"
    if encoder_size != "default":
        model_dir += f"_{encoder_size}"
    if freeze_encoder:
        model_dir += "_freeze"
    if freeze_feature_extractor:
        model_dir += "_freezeFE"
    if lora_encoder:
        model_dir += "_lora"
    if layer_selector_encoder is not None:
        model_dir += f"_layer{layer_selector_encoder}"
    return model_dir

def save_config(config, dir, filename):
    os.makedirs(dir, exist_ok=True)
    config_path = os.path.join(dir, filename)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    return config_path

def load_config(config_path):
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    return config

def patch_music2vec_config(model_id):
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    # remove the vocab size entry
    if "vocab_size" in config:
        del config["vocab_size"]
    return Data2VecAudioConfig(**config)
