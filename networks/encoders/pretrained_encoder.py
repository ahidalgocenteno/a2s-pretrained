
from torch import  stack, nn
from transformers import (
    AutoModel,
    AudioFlamingo3ForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
)
import torch
from peft import get_peft_model, LoraConfig
from my_utils.config import ENCODERS_CONFIG, DATASETS_CONFIG
from my_utils.helper import patch_music2vec_config


class PretrainedEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_size: str,
        ds_name: str,
        freeze_encoder: bool = False,
        freeze_feature_extractor: bool = False,
        lora: bool = False,
        lora_r: int = 8,
        embedding_dim: int = 256,
        layer_selection: int | None = None,
    ):
        super().__init__()

        # ── config validation ──────────────────────────────────────────────
        assert encoder_name in ENCODERS_CONFIG
        if encoder_size == "default":
            encoder_size = ENCODERS_CONFIG[encoder_name]["default_size"]
        assert encoder_size in ENCODERS_CONFIG[encoder_name]["supported_sizes"]
        assert not (freeze_encoder and freeze_feature_extractor), (
            "Cannot freeze both complete encoder and feature extractor."
        )
        assert not ((freeze_encoder or freeze_feature_extractor) and lora), (
            "LoRA requires trainable parameters; cannot be used with a frozen encoder."
        )
        # config extraction
        cfg = ENCODERS_CONFIG[encoder_name]
        self.requires_attention_mask = cfg.get("requires_attention_mask", False)
        self.encoder_name = encoder_name
        self.model_id     = cfg["model_id"][encoder_size]
        self.num_layers   = cfg["num_layers"][encoder_size]
        self.hidden_size  = cfg["embedding_dim"][encoder_size]
        self.lora         = lora
        # attention mask handling
        if self.requires_attention_mask:
            self._att_mask = self._get_attention_mask
            self.mask_argument = cfg.get("mask_argument", "attention_mask")
        else:
            self._att_mask = lambda x: {}
        # resampling factor for sequence length
        self.resampling_factor = (
            cfg["sampling_rate"] / DATASETS_CONFIG[ds_name]["sampling_rate"]
        ) if encoder_name != "cnn" else None
        # layer selection 
        assert (layer_selection is None or layer_selection == -1 or 1 <= layer_selection <= self.num_layers)
        self.layer_selection = layer_selection
        # aggregation / layer selection
        if layer_selection is None:
            self._g = self._aggregate_layers
            self.aggregator = nn.Sequential(
                nn.LayerNorm(self.num_layers),
                nn.Linear(self.num_layers, 1)
            )
        else:
            self._g = self._select_layer
        # hidden states extraction
        if encoder_name in ("flamingo3", "qwen2"):
            self._h = self._get_hidden_states_alm
        else:
            self._h = self._get_hidden_states 
        # sequence length after the encoder
        self.stride = cfg.get("stride", None)
        self.width  = cfg.get("width",  None)
        self.output_time_steps = cfg.get("output_time_steps", None)
        # load model
        self.pretrained_audio_encoder = self._load_model()
        # freezing and LoRA setup
        if freeze_encoder:
            self._freeze_all()
        elif freeze_feature_extractor:
            self._freeze_feature_extractor()
        if lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=0.1,
            )
            self.pretrained_audio_encoder = get_peft_model(self.pretrained_audio_encoder, lora_config)
            print("Using LoRA for audio encoder ////", end=" ")
            self.pretrained_audio_encoder.print_trainable_parameters()
        # final projection to match the desired embedding dimension
        self.projection = nn.Linear(self.hidden_size, embedding_dim)

    def _load_model(self):
        name = self.encoder_name
        if name == "wav2vec2":
            model = AutoModel.from_pretrained(self.model_id)
        elif name == "music2vec":
            config = patch_music2vec_config(self.model_id)
            model = AutoModel.from_pretrained(self.model_id, config=config, trust_remote_code=True)
        elif name == "mert":
            model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        elif name == "whisper":
            whisper = AutoModel.from_pretrained(self.model_id)
            model = whisper.encoder
            del whisper
        elif name == "flamingo3":
            flamingo = AudioFlamingo3ForConditionalGeneration.from_pretrained(self.model_id)
            model = flamingo.audio_tower
            del flamingo
        elif name == "qwen2":
            qwen2 = Qwen2AudioForConditionalGeneration.from_pretrained(self.model_id)
            model = qwen2.audio_tower
            del qwen2
        else:
            raise NotImplementedError(name)
        
        if name in ("flamingo3", "qwen2"):
            self._feature_conv_out = [None]
            def _capture_embed(module, args):
                self._feature_conv_out[0] = args[0]
            model.layers[0].register_forward_pre_hook(_capture_embed)

        # LoRA hook for models without input embeddings
        if self.lora and not self._has_input_embeddings(model):
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.feature_extractor.register_forward_hook(make_inputs_require_grad)
            model.enable_input_require_grads = lambda: None

        return model

    @staticmethod
    def _has_input_embeddings(model) -> bool:
        try:
            model.get_input_embeddings()
            return True
        except NotImplementedError:
            return False
        
    def get_max_len_post_enc(self, max_audio_len: int) -> int:
        """
        Upper bound on xl for a given max_audio_len (samples at the DATASET rate).
        Used to size positional embeddings / buffers before training starts.
        """
        if self.encoder_name in ("music2vec", "wav2vec2", "mert"):
            return int(((max_audio_len * self.resampling_factor) - self.width) // self.stride)
        elif self.encoder_name in ("whisper", "qwen2", "flamingo3"):
            # Output is always constant regardless of input duration.
            return self.output_time_steps

    def _freeze_all(self):
        self.pretrained_audio_encoder.eval()
        for p in self.pretrained_audio_encoder.parameters():
            p.requires_grad = False

    def _freeze_feature_extractor(self):
        for name, module in self.pretrained_audio_encoder.named_modules():
            if "feature_extractor" in name or "conv" in name:
                for p in module.parameters():
                    p.requires_grad = False
        if self.encoder_name in ("whisper", "qwen2", "flamingo3"):
            for p in self.pretrained_audio_encoder.conv1.parameters():
                p.requires_grad = False
            for p in self.pretrained_audio_encoder.conv2.parameters():
                p.requires_grad = False

    def _get_attention_mask(self, x):
        return {self.mask_argument: torch.ones(x.shape[0], x.shape[2], dtype=torch.long, device=x.device)}
    
    def _get_hidden_states(self, encoder_out):
        h = stack(encoder_out.hidden_states)  # [L, B, T, D]
        h = h.permute(1, 2, 3, 0)
        return h

    def _get_hidden_states_alm(self, encoder_out):
        h = (self._feature_conv_out[0],) + encoder_out.hidden_states[:-1] # skip pooling and add conv features
        h = stack(h)  # [L, B, T, D]
        h = h.permute(1, 2, 3, 0)
        return h

    def _select_layer(self, h):
        return h[..., self.layer_selection]

    def _aggregate_layers(self, h):
        h = h.to(self.aggregator[0].weight.dtype)  # cast to match aggregator dtype
        return self.aggregator(h).squeeze(-1)    

    def forward(self, x):
        x = self.pretrained_audio_encoder(   # [L, B, T, D]
            x.squeeze(1),
            output_hidden_states=True,
            return_dict=True,
            **self._att_mask(x),
        )            
        x = self._g(self._h(x))     # [B, T, D]
        x = self.projection(x.to(self.projection.weight.dtype))
        return x
