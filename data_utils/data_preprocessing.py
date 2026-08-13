import torch
import torchaudio.transforms as T
import torch.nn.functional as F
import math
from transformers import Wav2Vec2FeatureExtractor, AutoProcessor
from my_utils.config import *

NUM_CHANNELS = 1
IMG_HEIGHT = NUM_FREQ_BINS = 195

# PADDING ..............................................................

def pad_batch_spectrogram(x):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x

def pad_batch_transcripts(x, dtype=torch.int32):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.type(dtype=dtype)
    return x

# BATCH PREPARATION ..............................................................

class BatchPreparation:
    def __init__(self, encoder_name, ds_name, test=False):
        assert encoder_name in ENCODERS_CONFIG.keys(), f"Encoder {encoder_name} not supported, choose one of {list(ENCODERS_CONFIG.keys())}."

        # Set dataset sampling rate (datasets with different sampling rates can be added here)
        encoder_cofig = ENCODERS_CONFIG[encoder_name]
        self.sr_ds = DATASETS_CONFIG[ds_name]["sampling_rate"]
        self.sr_encoder = encoder_cofig["sampling_rate"]

        self.resampling_factor = self.sr_encoder / self.sr_ds if encoder_name != "cnn" else None
        self.width = encoder_cofig.get("width", None)
        self.stride = encoder_cofig.get("stride", None)
        self.output_time_steps = encoder_cofig.get("output_time_steps", None)
        self.encoder_name = encoder_name
        self.encoder_type = "cnn" if encoder_name == "cnn" else "pretrained"
        if encoder_name != "cnn":
            self.resampler = T.Resample(orig_freq=self.sr_ds, new_freq=self.sr_encoder)
        # SSL and WSL models
        if encoder_name == "music2vec":
            self.processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
            self.preprocess_fn = self.preprocess_wav2vec
        elif encoder_name == "mert":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
            self.preprocess_fn = self.preprocess_wav2vec
        elif encoder_name == "wav2vec2":
            self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            self.preprocess_fn = self.preprocess_wav2vec
        elif encoder_name == "whisper":
            self.processor = AutoProcessor.from_pretrained("openai/whisper-small")
            self.preprocess_fn = self.preprocess_whisper
        # Audio Language Models
        elif encoder_name == "qwen2":
            qwen2_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            self.processor = qwen2_processor.feature_extractor
            self.preprocess_fn = self.preprocess_alm
        elif encoder_name == "flamingo3":
            flamingo_processor = AutoProcessor.from_pretrained("nvidia/audio-flamingo-3-hf")
            self.processor = flamingo_processor.feature_extractor
            self.preprocess_fn = self.preprocess_alm
        # CNN
        elif encoder_name == "cnn":
            self.processor = None
            self.preprocess_fn = self.preprocess_cnn
        else:
            raise ValueError(f"Unsupported encoder name: {encoder_name}, choose from {list(ENCODERS_CONFIG.keys())}.")

        self.test = test

    # memory len getters
    def get_xl_cnn(self, x):
        return [
            math.ceil(i.shape[0] / ENCODERS_CONFIG["cnn"]["height_reduction"]) * \
            math.ceil(i.shape[1] / ENCODERS_CONFIG["cnn"]["width_reduction"])
            for i in x
        ]

    def get_xl_pretrained(self, x):
        if self.encoder_name in ("music2vec", "wav2vec2", "mert"):
            return [int((i.shape[0] * self.resampling_factor - self.width) // self.stride) for i in x]
        elif self.encoder_name in ("whisper", "qwen2", "flamingo3"):
            return [self.output_time_steps] * len(x)

    # preprocessing 

    def preprocess_cnn(self, x):
        x = [i.unsqueeze(0) if i.dim() == 2 else i for i in x]
        return pad_batch_spectrogram(x)
     
    def preprocess_wav2vec(self, x):
        """Preprocessing for music2vec, MERT, and wav2vec2."""
        x = [self.resampler(i) for i in x]
        processor_output = self.processor(x, sampling_rate=self.sr_encoder, return_tensors="pt", padding=False)
        return processor_output.input_values.squeeze(0)

    def preprocess_whisper(self, x):
        """Preprocessing for Whisper."""
        outputs = [
            self.processor(
                self.resampler(i).squeeze().detach().cpu().numpy(),
                return_tensors="pt",
                sampling_rate=self.sr_encoder
            ).input_features.squeeze(0)
            for i in x
        ]
        return torch.stack(outputs, dim=0)

    def preprocess_alm(self, x):
        """Preprocessing for Audio Language Models (Qwen2-Audio and Flamingo3)."""
        outputs = [
            self.processor(
                raw_speech=self.resampler(i).squeeze().detach().cpu().numpy(),
                return_tensors="pt",
                sampling_rate=self.sr_encoder,
            )
            for i in x
        ]
        return torch.stack([output.input_features.squeeze(0) for output in outputs], dim=0)
    
    
    def __call__(self, batch):
        x, y, id = zip(*batch)
        xl = torch.tensor(getattr(self, f"get_xl_{self.encoder_type}")(x) , dtype=torch.int32)
        x = self.preprocess_fn(x)  # Call assigned preprocessing function

        if self.test:
            y = pad_batch_transcripts(y, dtype=torch.int64)
            return x, y, id
        
        y_in = pad_batch_transcripts([i[:-1] for i in y], dtype=torch.int64)
        y_out = pad_batch_transcripts([i[1:] for i in y], dtype=torch.int64)
        return x, xl, y_in, y_out
