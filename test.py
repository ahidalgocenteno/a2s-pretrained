import os
import gc
import fire
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from networks.model import A2STransformer
from data_utils.amt_datamodule import AMTDataModule
from my_utils.seed import seed_everything
from my_utils.helper import load_config
from my_utils.config import *

seed_everything(42, benchmark=False)


def test(
    model_checkpoint: str,
    test_composer: str | None = "all",      # Only used for Quartets dataset
    num_workers: int = 8,
    ds_dir: str | None = None,
):
    gc.collect()
    torch.cuda.empty_cache()

    if not os.path.exists(model_checkpoint):
        print(f"Checkpoint {model_checkpoint} does not exist.")
        return
    
    model_base_path = os.path.splitext(model_checkpoint)[0]
    model_base_path = model_base_path.split("-v")[0] # Remove versioning info if present
    data_config = load_config(model_base_path + "_data_config.pkl")    
    data_config['ds_dir'] = ds_dir if ds_dir is not None else data_config.get('ds_dir', None)
    transformer_config = load_config(model_base_path + "_transformer_config.pkl")
    test_predictions_path = os.path.join(model_base_path + "_test_predictions.pkl")
    if test_composer is not None and test_composer != "all" and data_config["ds_name"] == "quartets":
        test_predictions_path = os.path.join(model_base_path + f"-{test_composer}_test_predictions.pkl")
    
    print("TEST EXPERIMENT")
    print(f"\tDataset: {data_config['ds_name']}")
    print(f"\tEncoder: {transformer_config['encoder_name']}")
    print(f"\tEncoder size: {transformer_config['encoder_size']}")
    print(f"\tCheckpoint: {model_checkpoint}")


    # Init DataModule
    datamodule = AMTDataModule(
        batch_size=1,
        num_workers=num_workers,
        test_composer=test_composer,
        **data_config
    )
    datamodule.setup(stage="test")

    # Load LightningModule from checkpoint
    model = A2STransformer.load_from_checkpoint(
        checkpoint_path=model_checkpoint,
        **transformer_config,
        log_predictions=True,
        log_predictions_path=test_predictions_path,
    )
    model.freeze()

    # Trainer and WandB logger
    trainer = Trainer(
        logger=WandbLogger(
            project="a2s-pretrained-test",
            name=f"Test-{data_config['ds_name']}-{transformer_config['encoder_name']}",
            log_model=False,
            config={
                "ds_name": data_config["ds_name"],
                "encoder_name": transformer_config["encoder_name"],
                "encoder_size": transformer_config["encoder_size"],
                "encoder_layer_selector": transformer_config["layer_selection_encoder"],
                "model_checkpoint": model_checkpoint,
                "composer": test_composer if data_config["ds_name"] == "quartets" else "-",
                "train_composer": data_config["composer"] if data_config["ds_name"] == "quartets" else "-",
                "score_encoding": data_config["score_encoding"] if data_config["ds_name"] == "primus" else "-",
                "sax_type": data_config["sax_type"] if data_config["ds_name"] == "realsax" else "-",
            },
        ),
        precision="bf16-mixed" if not transformer_config.get("lora_encoder", False) else "bf16",
    )

    # Run test
    trainer.test(model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    fire.Fire(test)
