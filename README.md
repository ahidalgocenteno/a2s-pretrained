<p align='center'>
  <a href='https://praig.ua.es/'><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
</p>

<h1 align='center'>MIREX 2026 – Audio-to-Score Transcription submission (MERT-330M + Transformer decoder)</h1>


<p align='center'>
  <img src='https://img.shields.io/badge/python-3.11.12-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/static/v1?label=License&message=MIT&color=blue' alt='License'>
</p>

<p align='center'>
  <a href='#about'>About</a> •
  <a href='#mirex-2026-submission'>MIREX 2026 Submission</a> •
  <a href='#setup'>Setup</a> •
  <a href='#running-the-transcription'>Running</a> •
  <a href='#resources-declaration'>Resources</a> •
  <a href='#reproducing-the-paper'>Reproducing the paper</a> •
  <a href='#citations'>Citations</a> •
  <a href='#acknowledgments'>Acknowledgments</a>
</p>

## About

This repository contains the official implementation of **On the use of Pre-trained Audio Encoders for Audio-to-Score Music Transcription**, accepted at the 27th International Society for Music Information Retrieval Conference (ISMIR 2026). This repository additionally provides the **entry point used to participate in the MIREX 2026 Audio-to-Score Transcription challenge**, using the best-performing encoder of the paper: **MERT-v1-330M**.


<p align="center">
  <img src="method.png" alt="General method: During the Adaptation phase, a pre-trained encoder and a randomly initialized decoder are trained on the annotated dataset. For each adaptation strategy, the operator defines the adapted encoder parameters, yielding the adapted model g. Subsequently, during the Transcription phase, the adapted model maps an unseen recording x to its predicted symbolic sequence." >
<br>
</p>


## MIREX 2026 Submission

### Our system

- **Encoder:** pre-trained [`m-a-p/MERT-v1-330M`](https://huggingface.co/m-a-p/MERT-v1-330M), adapted on the Quartets dataset (adaptation strategy: **fine-tuning**).
- **Decoder:** Transformer decoder with cross-attention over the encoder output, trained from scratch.
- **Output:** autoregressive (greedy) generation of \*\*kern tokens, detokenized into a full multi-spine `**kern` file including the header.
- **Training data:** Quartets dataset only (train split). No additional data was used for training or adaptation. The MERT encoder was pre-trained by its authors on external music data.
- **Checkpoint:** hosted at [`ahidalgocenteno/a2s-pretrained`](https://huggingface.co/ahidalgocenteno/a2s-pretrained) (`checkpoints/quartets_mert_330M_transformer.ckpt`) and downloaded automatically on first run. The weights have been directly uploaded from the training weights output, as shown in the script [upload_checkpoints_to_hf.py](upload_checkpoints_to_hf.py).

### Input / output

- **Input:** a folder containing audio files (`.flac`, `.wav`, `.mp3`, `.ogg`). The expected format is 16-bit FLAC, 22,050 Hz, ~30 s; any other format is converted automatically (downmixed to mono and resampled to 24 kHz for MERT).
- **Output:** one `**kern` file per input audio file, written to the output directory with the same base name (`<audio_name>.krn`).

## Setup

### Docker

A [Dockerfile](Dockerfile) with all system and Python dependencies is provided (PyTorch 2.7.1, CUDA 12.8, cuDNN 9):

```bash
docker build -t a2s-pretrained-image .
```

### Model weights and internet access

On first run, the script downloads from the Hugging Face Hub the A2S checkpoint and configuration from `ahidalgocenteno/a2s-pretrained`.

If you need to run **offline**, run the script once with internet access to populate the cache (`~/.cache/huggingface`, or the folder set in `HF_HOME`), then set `HF_HUB_OFFLINE=1`. If you use Docker, mount a persistent cache folder (`-v ~/.cache/huggingface:/root/.cache/huggingface`) to avoid re-downloading.

## Running the transcription

The entry point is `transcribe.py`. It takes a folder of audio files and writes one `**kern` file per audio.

```bash
python transcribe.py \
  --audio_dir /path/to/audio \
  --output_dir /path/to/output
```

| Argument | Default | Description |
|---|---|---|
| `--audio_dir` | – | Folder containing the audio files to transcribe. |
| `--output_dir` | `./transcriptions` | Folder where the `**kern` files are written (created if missing). |

Example output:

```
TRANSCRIBING AUDIO
	Encoder: mert
	Encoder size: 330M
	Checkpoint: .../quartets_mert_330M_transformer.ckpt
/path/to/audio/quartet_001.flac -> /path/to/output/quartet_001.krn
```

### With Docker

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/audio:/data/input:ro \
  -v /path/to/output:/data/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  a2s-pretrained-image \
  python transcribe.py --audio_dir /data/input --output_dir /data/output
```

No manual intervention is needed during execution. The script is deterministic (seed fixed to 42, greedy decoding).

## Resources declaration

As required by the MIREX 2026 A2S task:

| Item | Value |
|---|---|
| Training data size | Quartets train split — **24,351  excerpts** |
| Additional training data | MERT-v1-330M was pre-trained externally by its authors through large scale pre-training. The encoder-decoder setup is adapted with Quartets training set (see [train.py](train.py)) |
| Number of parameters | **322 M** |
| Training compute | **7d 17h 1m 57s on NVIDIA GeForce RTX 5090, 32GB** |
| Inference time on the evaluation set | **6h 17m 11s on NVIDIA GeForce RTX 5090, 32GB** |
| Peak GPU memory at inference | **22,65 GB** |
| Hardware limits (≤ 32 GB VRAM, ≤ 24 h on a V100) | Within limits |

## Reproducing the training

The following instructions reproduce the experiments of the paper (adaptation of all encoders, evaluation with MUSTER / MV2H).

### Install MUSTER

```bash
wget https://github.com/TimFelixBeyer/amtevaluation.github.io/raw/refs/heads/main/MUSTER_v231108.zip -O muster.zip
unzip muster.zip -d MUSTER_tmp
mv "MUSTER_tmp/MUSTER 2" MUSTER
rm -rf muster.zip MUSTER_tmp

cd MUSTER
./compile.sh
cd ..
```

### Authentication

Training logs to Weights & Biases and pulls models and datasets from the Hugging Face Hub:

```bash
wandb login
hf auth login
```

### Training (adaptation phase)

`train.py` adapts a pre-trained encoder, with options to freeze it, apply LoRA, or fully fine-tune (default):

```bash
python train.py --encoder_name <'cnn', 'music2vec', 'mert', 'wav2vec2', 'whisper', 'qwen2' or 'flamingo3'>  \
  <--freeze_encoder or --lora_encoder>
```

Models and configurations are saved at `./weights/<dataset>_<encoder name>_<adaptation>`. To adapt all evaluated encoders for a given strategy:

```bash
bash run_experiments.sh <lora, frozen or finetuning>
```

### Testing (transcription phase)

`test.py` transcribes the unseen test partition and computes the metrics:

```bash
python test.py --checkpoint_dir "./weights/<dataset>_<encoder name>_<adaptation>/all.ckpt" --test_composer <'all', 'mozart', 'beethoven' or 'haydn'>
```

## Citations

```bibtex
  TBD
```

## Acknowledgments

This research was supported by the Spanish Ministry of Science and Innovation through the LEMUR research project (PID2023-148259NB-I00), funded by MCIU/AEI/10.13039/501100011033/FEDER, EU, and the European Social Fund Plus (FSE+).

<p align="center">
    <a href="https://praig.ua.es/category/research/projects/"><img src="https://raw.githubusercontent.com/lemur-project/acknowledgments/refs/heads/main/infographics/lemur_logo.png" alt="LEMUR logo" height="60"></a>
    &nbsp;
    <a href="https://www.aei.gob.es/"><img src="https://raw.githubusercontent.com/lemur-project/acknowledgments/refs/heads/main/infographics/acknowledgements.png" alt="Ministry Logo, European Union Flag and Statal Research Agency Logo" height="60"></a>
    <br>
</p>