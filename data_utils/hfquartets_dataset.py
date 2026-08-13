import json
import os
import librosa
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from datasets import load_dataset
import soundfile as sf
import io
from datasets import Audio

from data_utils.encoding_convertions import krnParser
from my_utils.config import *
BAD_AUDIO_FILES = {"k590-02.010.flac"}

class HFQuartetsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        input_modality: str,
        partition: str,
        composer: str="all",
        test_composer=None,
    ):
        super().__init__()
        
        self.input_modality = input_modality
        self.sr = DATASETS_CONFIG["hfquartets"]["sampling_rate"]

        test_composer = composer if test_composer is None else test_composer
        if composer not in ["all", "haydn", "beethoven", "mozart"] or test_composer not in ["all", "haydn", "beethoven", "mozart"]:
            raise ValueError(f"Invalid composer '{composer}'. Valid options are ['all', 'haydn', 'beethoven', 'mozart'].")

        if partition == "train" and test_composer != composer:
            raise ValueError(f"Training partition cannot be used with a different test composer. Please set test_composer to None or the same as composer.")
        
        self.composer = composer if composer != "all" else "quartets"      
        test_composer = test_composer if test_composer != "all" else "quartets"

        # always the test composer, because if no test composer is provided it will be the same as the training composer
        # and if it is provided, it will be the one specified by the user
        self.hf_dataset = load_dataset(
            f"PRAIG/{test_composer}-quartets", 
            split=partition,
        )
        self.hf_dataset = self.patch_corrupt_audio(self.hf_dataset)

        # just to store metadata, the data itself comes from the Hugging Face dataset
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        self.parser = krnParser()
        # the vocab always comes from the training composer, not the testing one
        vocab = self._create_vocab(
            vocab_path=os.path.join(root_dir, "vocabs", f"vocab_{self.composer}.txt")
        )
        self.w2i, self.i2w = self.obtain_dictionaries(vocab)
        self.len_log_path = os.path.join(root_dir, "lens", f"max_lengths_{self.composer}.json")

    # UTILS -----------------------------------------------------------------

    def patch_corrupt_audio(self, ds):
        ds = ds.cast_column("audio", Audio(decode=False))

        def patch_bad_sample(example):
            if example["audio"]["path"] in BAD_AUDIO_FILES:
                arr, sr = sf.read(io.BytesIO(example["audio"]["bytes"]))
                buf = io.BytesIO()
                sf.write(buf, arr, sr, format="FLAC")
                example["audio"] = {"bytes": buf.getvalue(), "path": example["audio"]["path"]}
            return example

        ds = ds.map(patch_bad_sample)
        ds = ds.cast_column("audio", Audio(decode=True))
        return ds
                    
    # vocab utils

    def obtain_dictionaries(self, vocab):
        w2i = {}
        i2w = {}

        vocab = [SOS_TOKEN, EOS_TOKEN] + vocab
        vocab = sorted(set(vocab))
        
        # Add padding token at index 0
        w2i = {word: idx + 1 for idx, word in enumerate(vocab)}  
        w2i[PAD_TOKEN] = 0  # Assign 0 to padding
        
        i2w = {idx: word for word, idx in w2i.items()}

        return w2i, i2w

    def _extract_vocab(self):
        full_ds = load_dataset(f"PRAIG/{self.composer}-quartets")

        vocab = []
        for partition in ["train", "val", "test"]:
            for text in full_ds[partition]["transcript"]:
                transcript = self.parser.convert(text=text)
                vocab.extend(transcript)
        vocab = sorted(set(vocab))
        vocab_hist = Counter(vocab)

        return vocab, vocab_hist

    def _save_vocab(self, vocab_file_path, vocab):
        try:
            with open(vocab_file_path, "w") as f:
                for word in vocab:
                    f.write(word + "\n")
        except Exception as e:
            raise ValueError(f"Failed to save vocabulary to {vocab_file_path}: {e}")

        return vocab_file_path

    def _retrieve_vocab(self, vocab_file_path):
        if not os.path.exists(vocab_file_path):
            return None
        else:
            with open(vocab_file_path, "r") as f:
                vocab = f.read().splitlines()
            return vocab

    def _create_vocab(self, vocab_path):
        if not os.path.exists(vocab_path):
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            vocab, _ = self._extract_vocab()
            self._save_vocab(vocab_path, vocab)
        else:
            vocab = self._retrieve_vocab(vocab_path)
        return vocab


    #  len utils

    def get_max_lengths(self):  
        if os.path.exists(self.len_log_path):
            with open(self.len_log_path, "r") as f:
                lengths = json.load(f)
        else:
            os.makedirs(os.path.dirname(self.len_log_path), exist_ok=True)
            
            max_seq_len = 0
            max_width_audio, max_width_spectrogram = 0, 0

            full_ds = load_dataset("PRAIG/quartets-quartets")
            for partition in ["train", "val", "test"]:
                full_ds[partition] = self.patch_corrupt_audio(full_ds[partition])
                for sample in full_ds[partition]:
                    
                    score = self.parser.convert(text=sample["transcript"])
                    waveform = self._get_waveform(sample)
                    spec = self._get_spectrogram(sample)

                    max_seq_len = max(max_seq_len, len(score))
                    max_width_audio = max(max_width_audio, waveform.shape[1])
                    max_width_spectrogram = max(max_width_spectrogram, spec.shape[2])

            with open(self.len_log_path, "w") as f:
                lengths = {
                    "max_seq_len": max_seq_len,
                    "max_width_audio": max_width_audio,
                    "max_width_spectrogram": max_width_spectrogram
                }
                json.dump(lengths, f, indent=4)

        return lengths["max_seq_len"] + 1, lengths[f"max_width_{self.input_modality}"] 

    def _preprocess_transcript(self, transcript):        
        y = self.parser.convert(text=transcript)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]

        return torch.tensor(y, dtype=torch.int64)

    def _get_spectrogram(self, sample):
        audio_array = sample["audio"]["array"]
        fs = sample["audio"]["sampling_rate"]

        stft_fmax = 2093
        stft_frequency_filter_max = librosa.fft_frequencies(sr=fs, n_fft=2048) <= stft_fmax

        stft = librosa.stft(audio_array, hop_length=512, win_length=2048, window="hann")
        stft = stft[stft_frequency_filter_max]

        stft_db = librosa.amplitude_to_db(np.abs(np.array(stft)), ref=np.max)
        log_stft = ((1.0 / 80.0) * stft_db) + 1.0

        x = np.expand_dims(log_stft, axis=0)  # Add channel dimension
        x = torch.tensor(x, dtype=torch.float32)
        return x

    def _get_waveform(self, sample):
        audio_array = sample["audio"]["array"]
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        return waveform


    def __getitem_spectrogram__(self, idx):
        return self._get_spectrogram(self.hf_dataset[idx])

    def __getitem_audio__(self, idx):
        return self._get_waveform(self.hf_dataset[idx])

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        transcript = sample["transcript"]

        score = self._preprocess_transcript(transcript)
        modality_data = getattr(self, f"__getitem_{self.input_modality}__")(idx)

        return *modality_data, score, idx

    def __len__(self):
        return len(self.hf_dataset)