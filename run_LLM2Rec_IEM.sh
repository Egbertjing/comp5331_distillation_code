# Second stage of training LLM2Rec -- Item Embedding Modeling.

model_path="/home/yingzhi/huggingface_data/hub/Qwen2-0.5B"  # Replace with your own model path

# Stage 2 - Train MNTP
echo "Starting Stage 2 - Train MNTP..."
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 ./llm2rec/run_mntp.py ./llm2rec/train_mntp_config.json

# Stage 3 - Train SimCSE
echo "Starting Stage 3 - Train SimCSE..."
cp ${model_path}/*token* ./output/iem_stage1/Qwen2-0.5B-AmazonMix6-CSFT/checkpoint-100/
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29502 ./llm2rec/run_unsupervised_SimCSE.py ./llm2rec/train_simcse_config.json
