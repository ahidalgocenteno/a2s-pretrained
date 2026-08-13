import random
import torch
import warnings
import pickle
import os
from lightning.pytorch import LightningModule
from torch.nn import CrossEntropyLoss

from networks.encoders.cnn_encoder import CNNEncoder
from networks.encoders.pretrained_encoder import PretrainedEncoder

from networks.transformer_decoder import Decoder
from my_utils.pe import PositionalEncoding2D, PositionalEncoding1D
from my_utils.config import *
from my_utils.metrics import compute_metrics
from data_utils.data_preprocessing import IMG_HEIGHT

class A2STransformer(LightningModule):
    def __init__(
        self,
        max_seq_len,
        max_audio_len,
        w2i,
        i2w,
        ds_name,
        encoder_name,
        encoder_size,
        attn_window=-1,
        teacher_forcing_prob=0.5,
        freeze_encoder=False,
        freeze_feature_extractor=False,
        lora_encoder=False,
        layer_selection_encoder=None,
        learning_rate=1e-4,
        log_predictions_path=None,
        log_predictions=False,
    ):
        super(A2STransformer, self).__init__()
        self.save_hyperparameters(ignore=['w2i', 'i2w'])

        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = i2w
        self.padding_idx = w2i[PAD_TOKEN]
        self.max_seq_len = max_seq_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.learning_rate = learning_rate

        # Encoder + Positional Encoding
        if encoder_name == "cnn":
            assert encoder_size == "default", "CNNEncoder does not support different sizes"
            assert not (freeze_encoder or freeze_feature_extractor), "CNNEncoder cannot be frozen, it has to be trained from scratch"
            if layer_selection_encoder is not None:
                warnings.warn(
                    "layer_selection_encoder is ignored when using CNNEncoder",
                    UserWarning,
                    stacklevel=2,
                )
            if lora_encoder:
                warnings.warn(
                    "lora_encoder ignored when using CNNEncoder",
                    UserWarning,
                    stacklevel=2,
                )

            self.encoder = CNNEncoder(in_channels=1)
            max_height_post_enc, max_width_post_enc = self.encoder.get_max_dims_post_enc(
                img_height=IMG_HEIGHT,
                max_audio_len=max_audio_len,
            )
            self.pos_enc = PositionalEncoding2D(
                num_channels=256,
                max_height=max_height_post_enc,
                max_width=max_width_post_enc,
            )
        else:
            self.encoder = PretrainedEncoder(
                encoder_name=encoder_name,
                encoder_size=encoder_size,
                ds_name=ds_name,
                freeze_encoder=freeze_encoder,
                freeze_feature_extractor=freeze_feature_extractor,
                lora=lora_encoder,
                embedding_dim=256,
                layer_selection=layer_selection_encoder,
            )
            self.pos_enc = PositionalEncoding1D(
                max_len=self.encoder.get_max_len_post_enc(max_audio_len=max_audio_len),
                emb_dim=256,
            )

        self.decoder = Decoder(
            output_size=len(self.w2i),
            max_seq_len=max_seq_len,
            num_embeddings=len(self.w2i),
            padding_idx=self.padding_idx,
            attn_window=attn_window,
        )

        self.compute_loss = CrossEntropyLoss(ignore_index=self.padding_idx)

        self.log_predictions = log_predictions
        self.log_predictions_path = log_predictions_path
        self.retrieve_logged_test = False
        self.pred_log_dict = {}

        self.Y = []
        self.YHat = []
        self.Bids = []

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, amsgrad=False
        )

    def forward(self, x, xl, y_in):
        x = self.encoder(x)
        x = self.pos_enc(x)
        # flatten cnn features and permute
        if len(x.shape) == 4:
            x = x.flatten(2).permute(0, 2, 1).contiguous()

        y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=xl)
        return y_out_hat

    def apply_teacher_forcing(self, y):
        # y.shape = [batch_size, seq_len]
        y_errored = y.clone()
        # Create a random mask with the same shape as y_errored
        random_mask = torch.rand_like(y_errored, dtype=torch.float) < self.teacher_forcing_prob
        # Create a mask for non-padding tokens
        non_padding_mask = y != self.padding_idx
        # Combine the random mask and non-padding mask
        combined_mask = random_mask & non_padding_mask
        # Generate random indices for the entire matrix
        random_indices = torch.randint(0, len(self.w2i), y_errored.shape, device=y_errored.device)
        # Apply the random indices only where the combined mask is True
        y_errored = torch.where(combined_mask, random_indices, y_errored)
        return y_errored

    def training_step(self, batch, batch_idx):
        x, xl, y_in, y_out = batch
        y_in = self.apply_teacher_forcing(y_in)
        yhat = self.forward(x=x, xl=xl, y_in=y_in)
        loss = self.compute_loss(yhat, y_out)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y, id = batch
        if self.retrieve_logged_test and id[0] in self.pred_log_dict.keys():
            pred_log = self.pred_log_dict[id[0]]
            y = pred_log["y_true"]
            yhat = pred_log["y_pred"]
            id = id[0]
        else:
            # Encoder
            x = self.encoder(x=x)
            # Prepare for decoder
            # PE + flatten + permute
            if len(x.shape) == 4: # CCN encoder 
                x = self.pos_enc(x)
                x = x.flatten(2).permute(0, 2, 1).contiguous()
            else:
                x = self.pos_enc(x)
            # Autoregressive decoding
            y_in = torch.tensor([self.w2i[SOS_TOKEN]]).unsqueeze(0).long().to(x.device)
            yhat = []
            for _ in range(self.max_seq_len):
                y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=None)
                y_out_hat = y_out_hat[0, :, -1]  # Last token
                y_out_hat_token = y_out_hat.argmax(dim=-1).item()
                y_out_hat_word = self.i2w[y_out_hat_token]
                yhat.append(y_out_hat_word)
                if y_out_hat_word == EOS_TOKEN:
                    break

                y_in = torch.cat(
                    [y_in, torch.tensor([[y_out_hat_token]]).long().to(x.device)], dim=1
                )

            # Decoded ground truth
            y = [self.ytest_i2w[i.item()] for i in y[0][1:]]  # Remove SOS_TOKEN
            id = id[0]

        # Append to later compute metrics
        self.Y.append(y)
        self.YHat.append(yhat)
        self.Bids.append(id)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.no_grad()
    def on_validation_epoch_end(self, name="val", print_random_samples=False):
        # Log predictions
        if self.log_predictions and (not os.path.exists(self.log_predictions_path)) and (name == "test"):
            pred_log_dict = {id : {"y_true": y_true, "y_pred": y_pred} for id, y_true, y_pred in zip(self.Bids, self.Y, self.YHat)}
            with open(self.log_predictions_path, 'wb') as f:
                pickle.dump(pred_log_dict, f)
        
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat, stage=name)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"EXAMPLE PREDICTION FROM {name.upper()} SET:")
            print(f"ID - {self.Bids[index]}")
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")

        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        self.Bids.clear()

        return metrics

    @torch.no_grad()
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)
    
    @torch.no_grad()
    def on_test_epoch_start(self):
        if self.log_predictions and os.path.exists(self.log_predictions_path):
            self.retrieve_logged_test = True
            with open(self.log_predictions_path, 'rb') as f:
                self.pred_log_dict = pickle.load(f)
        else:
            self.retrieve_logged_test = False
        