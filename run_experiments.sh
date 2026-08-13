#!/bin/bash

declare -A encoders
encoders=(
    ["music2vec"]="default"
    ["mert"]="95M"
    ["whisper"]="small"
    ["wav2vec2"]="base"
    ["qwen2"]="default"
    ["flamingo3"]="default"
)

adaptation_strategy="${1:-finetuning}"

layer_selection="None"
dataset="hfquartets"
composer="all"

case "$adaptation_strategy" in
    lora)
        freeze_encoder="True"
        lora_encoder="True"
        ;;
    frozen)
        freeze_encoder="True"
        lora_encoder="False"
        ;;
    finetuning)
        freeze_encoder="False"
        lora_encoder="False"
        ;;
    *)
        echo "Invalid adaptation strategy: $adaptation_strategy"
        echo "Usage: $0 [lora|frozen|finetuning]"
        exit 1
        ;;
esac

echo "Running training for CNN baseline"
python train.py \
    --ds_name "$dataset" \
    --encoder_name "cnn" \
    --composer "$composer" \
    --freeze_encoder "$freeze_encoder" \
    --lora_encoder "$lora_encoder"

for encoder in "${!encoders[@]}"; do
    for size in ${encoders[$encoder]}; do
        echo "Running training for encoder: $encoder, size: $size"
        python train.py \
            --ds_name "$dataset" \
            --encoder_name "$encoder" \
            --encoder_size "$size" \
            --composer "$composer" \
            --freeze_encoder "$freeze_encoder" \
            --lora_encoder "$lora_encoder" \
            --layer_selector_encoder "$layer_selection"
    done
done
