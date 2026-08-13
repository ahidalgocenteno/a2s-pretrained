import os
import pandas as pd
from data_utils.encoding_convertions import krnParser
from data_utils.amt_dataset import AMTDataset
from my_utils.config import *

class RealSaxDataset(AMTDataset):
    def __init__(
        self,
        root_dir: str,
        input_modality: str,
        partition: str,
        sax_type: str,
        audio_type: str="real",
    ):
        super().__init__(root_dir=root_dir)
        
        self.root_dir = root_dir
        self.input_modality = input_modality
        self.sr = DATASETS_CONFIG["realsax"]["sampling_rate"]

        if sax_type not in ["alto", "tenor"]:
            raise ValueError(f"Invalid Saxophone type '{sax_type}'. Valid options are ['alto', 'tenor'].")
        if audio_type not in ["real", "fluidsynth", "mididdsp"]:
            raise ValueError(f"Invalid audio type '{audio_type}'. Valid options are ['real', 'fluidsynth', 'mididdsp'].")
        self.sax_type = sax_type
        self.audio_type = audio_type
        
        self._validate_partition(partition)
        self.partition_ids = self._load_partition_ids(
            partition_file=os.path.join(self.root_dir, "partitions", f"{partition}.lst"),
            partition=partition
        )
        self.parser = krnParser()
        vocab = self._create_vocab(
            vocab_path = os.path.join(self.root_dir, "vocabs", f"vocab_{self.sax_type}.txt")
        )
        self.w2i, self.i2w = self.obtain_dictionaries(vocab)
        self.len_log_path = os.path.join(self.root_dir, "lens", f"max_lengths_{self.audio_type}_{self.sax_type}.json")

    # INTERNAL UTILS -----------------------------------------------------------------

    def _get_all_ids(self):
        files_data_path = os.path.join(self.root_dir, "Alto_Sax_Index.csv") # alto and tenor have the same file names (acting as ids)
        df = pd.read_csv(files_data_path, sep=";")
        return df["FILENAME"].tolist()

    def _create_vocab(self, vocab_path):
        if not os.path.exists(vocab_path):
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            vocab, _ = self._extract_vocab(os.path.join(self.root_dir, "krn", self.sax_type), self.parser, [".krn", ".skm"])
            self._save_vocab(vocab_path, vocab)
        else:
            vocab = self._retrieve_vocab(vocab_path)
        return vocab

    def _get_audio_path_from_id(self, id):
        return os.path.join(self.root_dir, self.audio_type, self.sax_type, f"{id}.wav")
    
    def _get_score_path_from_id(self, id):
        score_path = os.path.join(self.root_dir, "krn", self.sax_type, f"{id}.krn")
        if not os.path.exists(score_path):
            score_path = os.path.join(self.root_dir, "krn", self.sax_type, f"{id}.skm")
        return score_path