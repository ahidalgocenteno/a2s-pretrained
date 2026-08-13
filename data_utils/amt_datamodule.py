from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from data_utils.data_preprocessing import BatchPreparation
from data_utils.primus_dataset import PriMusDataset
from data_utils.quartets_dataset import QuartetsDataset
from data_utils.hfquartets_dataset import HFQuartetsDataset
from data_utils.realsax_dataset import RealSaxDataset
from my_utils.config import *

class AMTDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        encoder_name: str,
        ds_dir: str | None = None,  # In case the dataset is not in the default location, user can specify the directory
        batch_size: int = 1,
        num_workers: int = 0,
        score_encoding: str = "kern",       # Only used for Primus dataset
        composer: str = "all",              # Only used for Quartets dataset
        test_composer: str | None = None,   # Only used for Quartets dataset
        sax_type: str = "alto"              # Only used for RealSax dataset
    ):
        super().__init__()

        assert ds_name in DATASETS_CONFIG.keys(), f"Dataset '{ds_name}' not recognized. Valid options are {list(DATASETS_CONFIG.keys())}."

        self.ds_name = ds_name
        

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.score_encoding = score_encoding
        self.composer = composer
        self.test_composer = test_composer
        self.sax_type = sax_type
        self.encoder_name = encoder_name
        self.input_modality = ENCODERS_CONFIG[encoder_name]["input_modality"] # raw audio or spectrogram (depends on the encoder)

        if ds_dir is None:
            self.ds_dir = DATASETS_CONFIG[self.ds_name]['default_dir']
            print(f"Using default dataset directory ./{DATASETS_CONFIG[self.ds_name]['default_dir']} for dataset {self.ds_name}. If the dataset is not in the default location, please specify the ds_dir argument.")            
        else:
            self.ds_dir = ds_dir
            
        # To prevent executing setup() twice
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str):
        if self.ds_name == "primus":
            if self.composer != "all":
                raise ValueError("Composer filtering is not supported for Primus dataset.")
            if stage == "fit":
                self.train_ds = PriMusDataset(self.ds_dir, partition="train", input_modality=self.input_modality, score_encoding=self.score_encoding)
                self.val_ds = PriMusDataset(self.ds_dir, partition="val", input_modality=self.input_modality, score_encoding=self.score_encoding)
            elif stage == "test" or stage == "predict":
                self.test_ds = PriMusDataset(self.ds_dir, partition="test", input_modality=self.input_modality, score_encoding=self.score_encoding)
        elif self.ds_name == "quartets":
            if self.score_encoding != "kern":
                raise ValueError("Quartets dataset only supports 'kern' score encoding.")
            if stage == "fit":
                self.train_ds = QuartetsDataset(self.ds_dir, partition="train", input_modality=self.input_modality, composer=self.composer)
                self.val_ds = QuartetsDataset(self.ds_dir, partition="val", input_modality=self.input_modality, composer=self.composer)
            elif stage == "test" or stage == "predict":
                self.test_ds = QuartetsDataset(self.ds_dir, partition="test", input_modality=self.input_modality, composer=self.composer, test_composer=self.test_composer)
        elif self.ds_name == "hfquartets":
            if self.score_encoding != "kern":
                raise ValueError("HFQuartets dataset only supports 'kern' score encoding.")
            if stage == "fit":
                self.train_ds = HFQuartetsDataset(self.ds_dir, partition="train", input_modality=self.input_modality, composer=self.composer)
                self.val_ds = HFQuartetsDataset(self.ds_dir, partition="val", input_modality=self.input_modality, composer=self.composer)
            elif stage == "test" or stage == "predict":
                self.test_ds = HFQuartetsDataset(self.ds_dir, partition="test", input_modality=self.input_modality, composer=self.composer, test_composer=self.test_composer)
        elif self.ds_name == "realsax":
            if self.score_encoding != "kern":
                raise ValueError("RealSax dataset only supports 'kern' score encoding.")
            if stage == "fit":
                self.train_ds = RealSaxDataset(self.ds_dir, partition="train", input_modality=self.input_modality, sax_type=self.sax_type)
                self.val_ds = RealSaxDataset(self.ds_dir, partition="val", input_modality=self.input_modality, sax_type=self.sax_type)
            elif stage == "test" or stage == "predict":
                self.test_ds = RealSaxDataset(self.ds_dir, partition="test", input_modality=self.input_modality, sax_type=self.sax_type)
        else:
            raise ValueError(f"Dataset '{self.ds_name}' is not supported. Choose from {list(DATASETS_CONFIG.keys())}.")                                                         
        
    def train_dataloader(self):
        collator = BatchPreparation(encoder_name=self.encoder_name, ds_name=self.ds_name)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collator, persistent_workers=self.num_workers > 0)
    
    def val_dataloader(self):
        collator = BatchPreparation(encoder_name=self.encoder_name, ds_name=self.ds_name, test=True)
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collator, persistent_workers=self.num_workers > 0)
        
    def test_dataloader(self):
        collator = BatchPreparation(encoder_name=self.encoder_name, ds_name=self.ds_name, test=True)
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collator, persistent_workers=self.num_workers > 0)
    
    def predict_dataloader(self):
        return self.test_dataloader()

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w
        
    def get_max_lengths(self):
        try:
            max_seq_len, max_audio_len = self.train_ds.get_max_lengths()
        except AttributeError:
            max_seq_len, max_audio_len = self.test_ds.get_max_lengths()
        return max_seq_len, max_audio_len
    