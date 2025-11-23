#!/bin/bash

# Define model settings as tuples: (save_info, model_path, bidirectional)
models_list=(
    # "Qwen2-0.5B-Backbone /home/yingzhi/huggingface_data/hub/Qwen2-0.5B"
    "Qwen2-0.5B-LLM2Rec-IEM_step500 ./output/iem_stage2/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-500"
    "Qwen2-0.5B-LLM2Rec-IEM_step1000 ./output/iem_stage2/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-1000"
)

# Define datasets as tuples: (dataset_name, cuda_device)
datasets=(
    "Games_5core 0"
    "Arts_5core 1"
    "Movies_5core 0"
    "Sports_5core 1"
    "Baby_5core 0"
    "Goodreads 1"
)

# Loop over each dataset setting (parallelized)
for dataset_setting in "${datasets[@]}"
do
    (
        # Extract dataset and CUDA device
        dataset=$(echo $dataset_setting | awk '{print $1}')
        cuda_device=$(echo $dataset_setting | awk '{print $2}')

        extraction_method="title"

        # Ensure the item_info directory exists
        mkdir -p "./item_info/${dataset}"

        # Loop over models sequentially for each dataset
        for model_setting in "${models_list[@]}"
        do
            # Split model_setting into save_info, model_path, and bidirectional
            save_info=$(echo $model_setting | awk '{print $1}')
            model_path=$(echo $model_setting | awk '{print $2}')
            bidirectional=1

            # Extract embeddings
            CUDA_VISIBLE_DEVICES=$cuda_device python extract_llm_embedding.py --dataset=$dataset \
                --model_path=$model_path \
                --item_prompt_type=$extraction_method \
                --bidirectional=$bidirectional \
                --save_info=$save_info

            # Define hyperparameters for evaluation
            # Default hyperparameters setting for SASRec
            lr=1.0e-3
            wd=1.0e-4
            loss_type="ce"
            model="SASRec"
            dr=0.3

            # Default hyperparameters setting for GRU4Rec
            # lr=1.0e-4
            # wd=1.0e-4
            # loss_type="ce"
            # model="GRU4Rec"
            # dr=0.3

            run_id="CR"
            embs="./item_info/${dataset}/${save_info}_${extraction_method}_item_embs.npy"

            # Random port to avoid conflicts
            port=$((12000 + RANDOM % 1000))

            # Evaluate the model
            CUDA_VISIBLE_DEVICES=$cuda_device accelerate launch --main_process_port=$port repeated_evaluate_with_seqrec.py \
                --model=$model \
                --dataset=$dataset \
                --lr=$lr \
                --weight_decay=$wd \
                --embedding=$embs \
                --dropout=$dr \
                --loss_type=$loss_type \
                --run_id=$run_id

            echo "✅ Finished processing dataset: $dataset with model: $save_info on CUDA device: $cuda_device"

        done
    ) &  # Run each dataset in parallel across available GPUs
done

# Wait for all parallel dataset processes to finish
wait

echo "🚀 All dataset experiments completed!"
