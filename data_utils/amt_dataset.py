import os
import warnings
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from collections import Counter
import glob
import random
import json
import tqdm

from my_utils.config import *


# Base class for automatic music transcription datasets
class AMTDataset(Dataset, ABC):
    def __init__(self, root_dir):
        
        self.root_dir = root_dir
        # to be defined in subclasses
        self.input_modality = None
        self.sr = None
        self.len_log_path = None
        self.parser = None
        self.partition_ids = None
        self.w2i = None        
        self.i2w = None
        
    # VALIDATION -----------------------------------------------------------
        
    def _validate_partition(self, partition):
        if partition not in VALID_PARTITIONS:
            raise ValueError(f"Invalid partition '{partition}'. Valid options are {VALID_PARTITIONS}.")

    # VOCAB UTILS -------------------------------------------------

    def _extract_vocab(self, folder_dir, parser, score_formats=None, files=None):
        vocab = []
        vocab_hist = []
        print("Extracting vocabulary...")
        
        if files is None and score_formats is not None:
            files = []
            for score_format in score_formats:
                files.extend(glob.glob(os.path.join(folder_dir, f"**/*{score_format}"), recursive=True))
        elif files is not None and score_formats is None:
            files = [os.path.join(folder_dir, file) for file in files]
        else:
            raise ValueError("Either score_formats or files must be provided, but not both or neither.")
        
        for file in files:
            score = parser.convert(file)
            vocab.extend(score)
            vocab_hist.extend(score)
            vocab = list(set(vocab))
            
        vocab_hist = Counter(vocab_hist)

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

    # PARTITION UTILS -------------------------------------------------

    def _load_partition_ids(self, partition_file, partition):
        if not os.path.exists(partition_file):
            os.makedirs(os.path.join(self.root_dir, "partitions"), exist_ok=True)
            partitions_dict = self._create_partitions(all_ids=self._get_all_ids())
            self._save_partitions(partitions_dict)
            return partitions_dict[partition]
        else:
            with open(partition_file, "r") as f:
                return [line.strip() for line in f]
            
    def _create_partitions(self, all_ids, train_size=0.8, val_test_size=0.2, random_seed=42):
        random.seed(random_seed)
        random.shuffle(all_ids)
        train = all_ids[:int(len(all_ids) * train_size)]
        val = all_ids[int(len(all_ids) * train_size):int(len(all_ids) * (train_size + val_test_size))]
        test = all_ids[int(len(all_ids) * (train_size + val_test_size)):]

        print(f"Number of training samples: {len(train)}")
        print(f"Number of test samples: {len(test)}")
        print(f"Number of validation samples: {len(val)}")

        # save to train.lst val.lst test.lst
        for key, partition_ids in zip(["train", "val", "test"], [train, val, test]):
            with open(os.path.join(self.root_dir, "partitions", f"{key}.lst"), "w") as f:
                for file in partition_ids:
                    f.write(file + "\n") 
        print("Partitions saved!")

        return {"train": train, "val": val, "test": test}
    
    def _save_partitions(self, partitions_dict):
        partitions_dirs = {
            par : os.path.join(self.root_dir, "partitions", f"{par}.lst") for par in partitions_dict.keys()
        }
        for key, partition_ids in partitions_dict.items():
            with open(partitions_dirs[key], "w") as f:
                for file in partition_ids:
                    f.write(file + "\n") 
        print("Partitions saved!")

        return partitions_dirs
    
    # LEN UTILS -------------------------------------------------

    def get_max_lengths(self, debug=True):
        # check if the max lengths are already calculated
        if os.path.exists(self.len_log_path):
            with open(self.len_log_path, "r") as f:
                lengths = json.load(f)
        else:
            os.makedirs(os.path.dirname(self.len_log_path), exist_ok=True)
            # get the maximum sequence length and the maximum spectrogram width
            max_seq_len = 0
            max_width_audio, max_width_spectrogram = 0, 0
            # get the files from the train partition
            ds_ids = self._get_all_ids()
            print("Calculating max lengths ...")
            for id in (tqdm.tqdm(ds_ids, dynamic_ncols=True) if debug else ds_ids):
                # for this count, eos and sos are not included, eos is later added
                score_path = self._get_score_path_from_id(id)
                score = self.parser.convert(score_path)
                max_seq_len = max(max_seq_len, len(score))

                waveform = self.__getitem_audio__(id)
                spectrogram = self.__getitem_spectrogram__(id)

                max_width_audio = max(max_width_audio, waveform.shape[1])
                max_width_spectrogram = max(max_width_spectrogram, spectrogram.shape[2])
            
            # log the max lengths
            with open(self.len_log_path, "w") as f:
                lengths = {
                    "max_seq_len": max_seq_len, 
                    "max_width_audio": max_width_audio, 
                    "max_width_spectrogram": max_width_spectrogram
                }
                json.dump(lengths, f, indent=4)

        # take into account EOS token
        return lengths["max_seq_len"] + 1, lengths[f"max_width_{self.input_modality}"]

    # DATA PREPROCESSING ---------------------------------------------------------------

    def _preprocess_transcript(self, id: str):        
        path = self._get_score_path_from_id(id)

        y = self.parser.convert(path)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]

        return torch.tensor(y, dtype=torch.int64)

    def __getitem_audio__(self, id: str):
        wav_path = self._get_audio_path_from_id(id)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*torchaudio.load_with_torchcodec.*") 
            waveform, _ = torchaudio.load(wav_path)        
        return waveform

    def __getitem_audio_stereo__(self, id: str):
        wav_path = self._get_audio_path_from_id(id)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*torchaudio.load_with_torchcodec.*")
            waveform, _ = torchaudio.load(wav_path)        
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def __getitem_spectrogram__(self, id: str):
        wav_path = self._get_audio_path_from_id(id)
        y, fs = librosa.load(wav_path, sr=self.sr)
        stft_fmax = 2093
        stft_frequency_filter_max = librosa.fft_frequencies(sr=fs, n_fft=2048) <= stft_fmax

        stft = librosa.stft(y, hop_length=512, win_length=2048, window="hann")
        stft = stft[stft_frequency_filter_max]

        stft_db = librosa.amplitude_to_db(np.abs(np.array(stft)), ref=np.max)
        log_stft = ((1.0 / 80.0) * stft_db) + 1.0
        
        x = np.expand_dims(log_stft, 0)
        x = torch.from_numpy(x)  # [1, freq_bins, time_frames]
        return x
    
    # GETTERS ---------------------------------------------------------------
            
    def __getitem__(self, idx):
        id = self.partition_ids[idx]

        score = self._preprocess_transcript(id)
        modality_data = getattr(self, f"__getitem_{self.input_modality}__")(id)

        return *modality_data, score, id
    
    def __len__(self):
        return len(self.partition_ids)
    
    # ABSTRACT METHODS -----------------------------------------------------

    @abstractmethod
    def _get_all_ids(self):
        pass

    @abstractmethod
    def _get_audio_path_from_id(self, id):
        pass

    @abstractmethod
    def _get_score_path_from_id(self, id):
        pass

    @abstractmethod
    def _create_vocab(self):
        pass