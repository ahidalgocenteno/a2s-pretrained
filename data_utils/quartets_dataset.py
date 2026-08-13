import os
from data_utils.encoding_convertions import krnParser
from data_utils.amt_dataset import AMTDataset
from my_utils.config import *


class QuartetsDataset(AMTDataset):
    def __init__(
        self,
        root_dir: str,
        input_modality: str,
        partition: str,
        composer: str="all",
        test_composer=None,
    ):
        super().__init__(root_dir=root_dir)
        
        self.root_dir = root_dir
        self.input_modality = input_modality
        self.sr = DATASETS_CONFIG["quartets"]["sampling_rate"]

        test_composer = composer if test_composer is None else test_composer
        if composer not in ["all", "haydn", "beethoven", "mozart"] or test_composer not in ["all", "haydn", "beethoven", "mozart"]:
            raise ValueError(f"Invalid composer '{composer}'. Valid options are ['all', 'haydn', 'beethoven', 'mozart'].")
        if partition == "train" and test_composer != composer:
            raise ValueError(f"Training partition cannot be used with a different test composer. Please set test_composer to None or the same as composer.")
        
        self.composer = composer if composer != "all" else "quartets"      
        test_composer = test_composer if test_composer != "all" else "quartets"

        # this is called here because it needs the internal utils of the class
        self._validate_partition(partition)
        self.partition_ids = self._load_partition_ids(
            partition_file=os.path.join(self.root_dir, "partitions", test_composer, f"{partition}.txt"),
            partition=partition
        )
        self.parser = krnParser()
        vocab = self._create_vocab(
            vocab_path=os.path.join(self.root_dir, "vocabs", f"vocab_{self.composer}.txt")
        )
        self.w2i, self.i2w = self.obtain_dictionaries(vocab)
        self.len_log_path = os.path.join(self.root_dir, "lens", f"max_lengths_{self.composer}.json")
        

    # UTILS -----------------------------------------------------------------

    def _get_all_ids(self):
        files = []
        for par in ["train", "val", "test"]:
            partition_file = os.path.join(self.root_dir, "partitions", self.composer, f"{par}.txt")
            with open(partition_file, "r") as f:
                files.extend([line.strip() for line in f])
        return files

    def _create_vocab(self, vocab_path):
        if not os.path.exists(vocab_path):
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            files = [f"{id}.krn" for id in self._get_all_ids()]
            vocab, _ = self._extract_vocab(os.path.join(self.root_dir, "krn"), self.parser, files=files)
            self._save_vocab(vocab_path, vocab)
        else:
            vocab = self._retrieve_vocab(vocab_path)
        return vocab
    
    def _get_audio_path_from_id(self, id):
        return os.path.join(self.root_dir, "flac", f"{id}.flac")
    
    def _get_score_path_from_id(self, id):
        return os.path.join(self.root_dir, "krn", f"{id}.krn")

    