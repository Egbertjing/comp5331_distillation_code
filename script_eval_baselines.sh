#!/bin/bash

# Define datasets and their corresponding CUDA_VISIBLE_DEVICES
datasets=(
    "Games_5core 0"
    "Arts_5core 0"
    "Movies_5core 0"
    "Goodreads 1"
    "Sports_5core 1"
    "Baby_5core 1"
)

# Define the fixed parameters
run_id="Eval_Embeddings"
model="SASRec"
dr=0.3
port_base=12324
lr=1.0e-3
wd=1.0e-4
loss_type="ce"

# Run experiments for all datasets simultaneously
for dataset_entry in "${datasets[@]}"; do
    (
        # Split dataset_entry into dataset name and CUDA device
        IFS=' ' read -r dataset cuda_device <<< "$dataset_entry"

        # Define embeddings for the current dataset
        embeddings=(
            # "./item_info/${dataset}/BGE_title_item_embs.npy"
            # "./item_info/${dataset}/Blair_title_item_embs.npy"
            # "./item_info/${dataset}/EasyRec_title_item_embs.npy"
            # "./item_info/${dataset}/BERT_title_item_embs.npy"
            "./item_info/${dataset}/RoBERTa_large_sentence_title_item_embs.npy"
            # "./item_info/${dataset}/GTE_7B_title_item_embs.npy"
        )

        for embs in "${embeddings[@]}"; do
            echo "Running evaluation with dataset=$dataset, CUDA_VISIBLE_DEVICES=$cuda_device, and embeddings=$embs"
            CUDA_VISIBLE_DEVICES=$cuda_device accelerate launch --main_process_port=$((port_base + cuda_device)) repeated_evaluate_with_seqrec.py \
                --model=$model \
                --dataset=$dataset \
                --lr=$lr \
                --weight_decay=$wd \
                --embedding=$embs \
                --dropout=$dr \
                --loss_type=$loss_type \
                --run_id=$run_id
        done
    ) &
done

# Wait for all background processes to complete
wait
