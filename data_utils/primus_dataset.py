import os
from data_utils.encoding_convertions import krnParser, semanticAgnosticParser
from data_utils.amt_dataset import AMTDataset
from my_utils.config import *


class PriMusDataset(AMTDataset):
    def __init__(
        self,
        root_dir: str,
        input_modality: str,
        partition: str,
        score_encoding: str="kern",
    ):  
        super().__init__(root_dir=root_dir)

        self.root_dir = root_dir
        self.encoding = score_encoding
        self.input_modality = input_modality
        self.sr = DATASETS_CONFIG["primus"]["sampling_rate"]

        self._validate_partition(partition)
        self.partition_ids = self._load_partition_ids(
            partition_file=os.path.join(self.root_dir, "partitions", f"{partition}.lst"),
            partition=partition
        )
        self.parser, self.score_extension = self._get_parser(score_encoding)
        vocab = self._create_vocab(
            vocab_path=os.path.join(self.root_dir, "vocabs", f"vocab_{self.encoding}.txt")
        )
        self.w2i, self.i2w = self.obtain_dictionaries(vocab)
        self.len_log_path = os.path.join(self.root_dir, "lens", f"max_lengths_{self.encoding}.json")

    # INTERNAL UTILS -----------------------------------------------------------------
            
    def _get_all_ids(self):
        files_data_path = os.path.join(self.root_dir, "files.lst")
        with open(files_data_path, "r") as f:
            return [line.strip() for line in f]

    def _create_vocab(self, vocab_path):
        if not os.path.exists(vocab_path):
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            # for Primus, we have three score encodings: kern, agnostic, semantic
            vocab, _ = self._extract_vocab(self.root_dir, self.parser, [self.score_extension])
            self._save_vocab(vocab_path, vocab)
        else:
            vocab = self._retrieve_vocab(vocab_path)
        return vocab

    def _get_parser(self, score_encoding):
        if score_encoding == "kern":
            return krnParser(), "krn" # "kern" file format is .krn
        return semanticAgnosticParser(), score_encoding # "agnostic" and "semantic" file formats are .agnostic and .semantic

    def _get_audio_path_from_id(self, id):
        return os.path.join(self.root_dir, id, f"{id}.wav")
    
    def _get_score_path_from_id(self, id):
        return os.path.join(self.root_dir, id, f"{id}.{self.score_extension}")
