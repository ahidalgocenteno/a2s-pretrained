import os

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor

class AudioProcessor:
    """Prepare one audio file for the MERT encoder."""

    ENCODER_SAMPLING_RATE = 24000
    FEATURE_WIDTH = 80
    FEATURE_STRIDE = 320

    def __init__(self):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M"
        )

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, sample_rate

    def get_xl(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the MERT time dimension used by the decoder mask."""
        num_samples = waveform.shape[-1]
        xl = (num_samples - self.FEATURE_WIDTH) // self.FEATURE_STRIDE
        return torch.tensor([xl], dtype=torch.int32)

    def process(self, audio_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, sample_rate = self._load_audio(audio_path)
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=self.ENCODER_SAMPLING_RATE,
        )
        features = self.processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.ENCODER_SAMPLING_RATE,
            return_tensors="pt",
            padding=False,
        )
        return features.input_values, self.get_xl(waveform)