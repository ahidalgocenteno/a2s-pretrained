import os
import gc
import torch

import argparse
from networks.model import A2STransformer
from data_utils.audio_processor import AudioProcessor
from my_utils.seed import seed_everything
from my_utils.helper import load_config
from data_utils.encoding_convertions import krnParser
from huggingface_hub import hf_hub_download

SOS_TOKEN = "<SOS>"  # Start-of-sequence token
EOS_TOKEN = "<EOS>"  # End-of-sequence token
PAD_TOKEN = "<PAD>"  # Padding token

seed_everything(42, benchmark=False)

MODEL_CHECKPOINT = hf_hub_download(
    repo_id="ahidalgocenteno/a2s-pretrained",
    filename="checkpoints/quartets_mert_330M_transformer.ckpt",
)

MODEL_CONFIG = hf_hub_download(
    repo_id="ahidalgocenteno/a2s-pretrained",
    filename="checkpoints/quartets_mert_330M_transformer_config.pkl",
)


def transcribe_audio(
    audio_dir: str,   # path of the directory containing audio files to transcribe
    output_dir: str,  # path of the directory where transcriptions will be saved
):
    gc.collect()
    torch.cuda.empty_cache()

    model_checkpoint = MODEL_CHECKPOINT
    transformer_config = load_config(MODEL_CONFIG)
    
    print("TRANSCRIBING AUDIO")
    print(f"\tEncoder: {transformer_config['encoder_name']}")
    print(f"\tEncoder size: {transformer_config['encoder_size']}")
    print(f"\tCheckpoint: {model_checkpoint}")


    model = A2STransformer.load_from_checkpoint(
        checkpoint_path=model_checkpoint,
        **transformer_config,
        log_predictions=True,
    )
    model.freeze()
    model.eval()

    processor = AudioProcessor()
    device = next(model.parameters()).device

    audio_paths = []
    if audio_dir is not None:
        audio_paths.extend(
            os.path.join(audio_dir, filename)
            for filename in sorted(os.listdir(audio_dir))
            if filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg"))
        )
    if not audio_paths:
        raise ValueError("No audio files were found to transcribe.")

    output_dir = output_dir or os.path.join(
        os.path.dirname(audio_paths[0]), "transcriptions"
    )
    os.makedirs(output_dir, exist_ok=True)

    krn_parser = krnParser()

    for audio_path in audio_paths:
        x, xl = processor.process(audio_path)
        x = x.to(device)
        xl = xl.to(device)
        
        prediction = []
        with torch.no_grad():
            memory = model.encoder(x)
            memory = model.pos_enc(memory)

            y_in = torch.tensor(
                [[model.w2i[SOS_TOKEN]]], dtype=torch.long, device=device
            )

            y_in = torch.tensor(
                [[model.w2i[SOS_TOKEN]]], dtype=torch.long, device=device
            )
            yhat = []
            for _ in range(model.max_seq_len):
                y_out_hat = model.decoder(tgt=y_in, memory=memory, memory_len=None)
                y_out_hat = y_out_hat[0, :, -1]  # Last token
                y_out_hat_token = y_out_hat.argmax(dim=-1).item()
                y_out_hat_word = model.i2w[y_out_hat_token]
                yhat.append(y_out_hat_word)
                prediction.append(y_out_hat_word)
                if y_out_hat_word == EOS_TOKEN:
                    break

                y_in = torch.cat(
                    [y_in, torch.tensor([[y_out_hat_token]]).long().to(device)], dim=1
                )

        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(audio_path))[0]}.krn",
        )
        krn_parser.detokenize_to_string(
                token_sequence=prediction,
                n_voices = None,
                include_header = True,
                write_path = output_path,
            )
        print(f"{audio_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default=None, help="Path to the directory containing audio files to transcribe")
    parser.add_argument("--output_dir", type=str, default="./transcriptions", help="Path to the directory where transcriptions will be saved")
    args = parser.parse_args()

    assert args.audio_dir is not None, "The --audio_dir argument is required."

    transcribe_audio(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir
    )