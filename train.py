import gc
import fire
import torch
import os
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from networks.model import A2STransformer
from data_utils.amt_datamodule import AMTDataModule
from my_utils.seed import seed_everything
from my_utils.helper import get_model_dir, save_config

from my_utils.config import *

seed_everything(42, benchmark=False)

def train(
    encoder_name: str,
    ds_name: str = "hfquartets",
    encoder_size: str = "default",
    ds_dir: str | None = None,
    freeze_encoder: bool = False,
    freeze_feature_extractor: bool = False,
    lora_encoder:  bool = False,
    layer_selector_encoder: int | None = None,
    epochs: int = 500,
    patience: int = 5,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    num_workers: int = 8,
    check_val_every_n_epoch: int = 5,
    attention_window: int = 100,
    composer = "all",             # Only used for Quartets dataset
    score_encoding = "kern",      # Only used for Primus dataset
    sax_type = "alto",            # Only used for RealSax dataset
    resume_ckpt_path: str | None = None,
    run_id: str | None = None,
):
    gc.collect()
    torch.cuda.empty_cache()

    # Experiment info
    print("TRAIN EXPERIMENT")
    print(f"\tDataset: {ds_name}")
    print(f"\tEncoder: {encoder_name}")
    print(f"\tEncoder size: {encoder_size}")
    print(f"\tFreeze encoder: {freeze_encoder}")
    print(f"\tLayer selector encoder: {layer_selector_encoder}")
    print(f"\tLearning rate: {learning_rate}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tCheck Val Every N epoch: {check_val_every_n_epoch}")

    # SETUP DATA, MODEL CONFIGS AND DATAMODULE
    data_config = {
        "ds_name": ds_name,
        "ds_dir": ds_dir,
        "encoder_name": encoder_name,
        "composer": composer,
        "sax_type": sax_type,
        "score_encoding": score_encoding,
    }

    datamodule = AMTDataModule(**data_config, num_workers=num_workers, batch_size=batch_size)
    datamodule.setup(stage="fit")
    w2i, i2w = datamodule.get_w2i_and_i2w()    
    max_seq_len, max_audio_len = datamodule.get_max_lengths()
    
    encoder_size = ENCODERS_CONFIG[encoder_name]["default_size"] if encoder_size == "default" else encoder_size
    a2s_transformer_config = {
        "ds_name": ds_name,
        "encoder_name": encoder_name,
        "encoder_size": encoder_size,
        "max_seq_len": max_seq_len,
        "max_audio_len": max_audio_len,
        "w2i": w2i,
        "i2w": i2w,
        "attn_window": attention_window,
        "teacher_forcing_prob": 0.2,
        "learning_rate": learning_rate,
        "layer_selection_encoder": layer_selector_encoder,
        "freeze_encoder": freeze_encoder,
        "freeze_feature_extractor": freeze_feature_extractor,
        "lora_encoder": lora_encoder,
    }

    # CALLBACKS AND MODEL CHECKPOINTS
    model_dir = get_model_dir(data_config, a2s_transformer_config)
    model_filename = getattr(datamodule, DATASETS_CONFIG[ds_name]["filename"])
    save_config(a2s_transformer_config, model_dir, model_filename + "_transformer_config.pkl")
    save_config(data_config, model_dir, model_filename + "_data_config.pkl")

    best_checkpoint = ModelCheckpoint(
        dirpath=model_dir,
        filename=model_filename,
        monitor="val_sym-er",
        verbose=True,
        save_last=False,
        save_top_k=1,
        save_weights_only=False,
        mode="min", # save the model with the lowest validation sym-er
        auto_insert_metric_name=False,
        every_n_epochs=5,
        save_on_train_epoch_end=False,
    )
    last_checkpoint = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_filename}_last",
        monitor="epoch", # save with the highest epoch number
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    callbacks = [
        best_checkpoint,
        last_checkpoint,
        EarlyStopping(
            monitor="val_sym-er",
            min_delta=0.01,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=600.00,
            check_on_train_epoch_end=False,
        ),
    ]

    # WANDB LOGGER CONFIGURATION
    wandb_config = {
        "project": "a2s-pretrained",
        "group": f"{ds_name}_{encoder_name}",
        "name": f"Train-{ds_name}-{encoder_name}_Test-{ds_name}-{encoder_name}",
        "log_model": True,
        "id": run_id,
        "config": {
            "encoder_name": encoder_name,
            "encoder_size": a2s_transformer_config["encoder_size"],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "ds_name": ds_name,
            "composer": composer if ds_name == "quartets" else "-",
            "score_encoding": score_encoding if ds_name == "primus" else "-",
            "sax_type": sax_type if ds_name == "realsax" else "-",
            "encoder_layer_selector": "aggregate" \
                if layer_selector_encoder is None else layer_selector_encoder,
        },
    }
    # MODEL INITIALIZATION / RESUME FROM CHECKPOINT
    if resume_ckpt_path:
        wandb_config["id"] = run_id
        wandb_config["resume"] = "must" if run_id is not None else "allow"
        model = A2STransformer.load_from_checkpoint(
            checkpoint_path=resume_ckpt_path, 
            **a2s_transformer_config
        )
    else:
        model = A2STransformer(**a2s_transformer_config)
    
    # TRAINING EXPERIMENT
    trainer = Trainer(
        logger=WandbLogger(**wandb_config),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        deterministic=False,
        benchmark=False,
        precision="bf16-mixed" if not lora_encoder else "bf16",
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path) 
    
    # TESTING EXPERIMENT
    model = A2STransformer.load_from_checkpoint(
        checkpoint_path=best_checkpoint.best_model_path,
        **a2s_transformer_config,
        log_predictions=True,
        log_predictions_path=os.path.join(model_dir, model_filename + "_test_predictions.pkl"),
    )

    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)