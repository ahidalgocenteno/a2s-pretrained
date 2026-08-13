SOS_TOKEN = "<SOS>"  # Start-of-sequence token
EOS_TOKEN = "<EOS>"  # End-of-sequence token
PAD_TOKEN = "<PAD>"  # Padding token

VALID_MODALITIES = {"audio", "spectrogram"}
VALID_PARTITIONS = {"train", "val", "test"}

DATASETS_CONFIG = {
    "primus": {
        "sampling_rate": 22050,
        "filename" : "score_encoding",
        "default_dir": "data/Primus",
    },
    "quartets": {
        "sampling_rate": 22050,
        "filename" : "composer",
        "default_dir": "data/Quartets",
    },
    "hfquartets" : {
        "sampling_rate": 22050,
        "filename" : "composer",
        "default_dir": "data/HF-Quartets",
    },
    "realsax": {
        "sampling_rate": 44100,
        "filename" : "sax_type",
        "default_dir": "data/RealSax",
    },
}

ENCODERS_CONFIG = {
    # SOTA : CNN
    "cnn": {
        "input_modality": "spectrogram",
        "sampling_rate": None,
        "height_reduction": 16,
        "width_reduction": 8,
        "default_size": "default",
        "supported_sizes": ["default"],
    },
    
    #  AUDIO UNDERSTANDING MODELS - WEAKLY AND SELF SUPERVISED
    "music2vec": {
        "input_modality": "audio",
        "sampling_rate": 16000,
        "requires_attention_mask": False,
        "default_size": "default",
        "stride": 320,
        "width": 80,
        "supported_sizes": ["default"],
        # parameters size-dependent
        "num_layers": {
            "default": 13, # layers = 1 (feature extractor) + N transformer layers, same for all the encoders
        },
        "embedding_dim": {
            "default": 768,
        },
        "model_id": {
            "default": "m-a-p/music2vec-v1",
        },
    },
    "mert": {
        "input_modality": "audio",
        "sampling_rate": 24000,
        "requires_attention_mask": False,
        "stride": 320,
        "width": 80,
        "default_size": "95M",
        "supported_sizes": ["95M", "330M"],
        "num_layers": {
            "95M": 13,
            "330M": 25,
        },
        "embedding_dim": {
            "95M": 768,
            "330M": 1024,
        },
        "model_id": {
            "95M": "m-a-p/MERT-v1-95M",
            "330M": "m-a-p/MERT-v1-330M",
        },
    },
    "wav2vec2": {
        "input_modality": "audio",
        "sampling_rate": 16000,
        "requires_attention_mask": False,
        "stride": 320,
        "width": 80,
        "default_size": "base",
        "supported_sizes": ["base", "large"],
        "num_layers": {
            "base": 13,
            "large": 25,
        },
        "embedding_dim": {
            "base": 768,
            "large": 1024,
        },
        "model_id": {
            "base": "facebook/wav2vec2-base",
            "large": "facebook/wav2vec2-large",
        },
    },
    "wav2vec2-audioset" : {
        "input_modality": "audio",
        "sampling_rate": 16000,
        "requires_attention_mask": False,
        "stride": 320,
        "width": 80,
        "default_size": "base",
        "supported_sizes": ["base", "large"],
        "num_layers": {
            "base": 13,
            "large": 25,
        },
        "embedding_dim": {
            "base": 768,
            "large": 1024,
        },
        "model_id": {
            "base": "ALM/wav2vec2-base-audioset",
            "large": "ALM/wav2vec2-large-audioset",
        },
    },
    "whisper": {
        "input_modality": "audio",
        "default_size": "small",
        "sampling_rate": 16000,
        "requires_attention_mask": False,
        "output_time_steps": 1500,
        "supported_sizes": ["base", "small", "medium", "large"],
        "num_layers": {
            "base": 7,
            "small": 13,
            "medium": 25,
            "large": 33,
        },
        "embedding_dim": {
            "base": 512,
            "small": 768,
            "medium": 1024,
            "large": 1280,
        },
        "model_id": {
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v3",
        },
    },
    # AUDIO LANGUAGE MODELS - LARGE MODELS WITH INSTRUCTION TUNING
    "qwen2": {
        "input_modality": "audio",
        "default_size": "default",
        "sampling_rate": 16000,
        "requires_attention_mask": False,
        "output_time_steps": 1500,
        "supported_sizes": ["default"],
        "num_layers": {
            "default": 33,
        },
        "embedding_dim": {
            "default": 1280,
        },
        "model_id": {
            "default": "Qwen/Qwen2-Audio-7B-Instruct",
        },
    },
    "flamingo3": {
        "input_modality": "audio",
        "default_size": "default",
        "sampling_rate": 16000,
        "requires_attention_mask": True,
        "mask_argument": "input_features_mask",
        "output_time_steps": 1500,
        "supported_sizes": ["default"],
        "num_layers": {
            "default": 33,
        },
        "embedding_dim": {
            "default": 1280,
        },
        "model_id": {
            "default": "nvidia/audio-flamingo-3-hf",
        },
    },
}