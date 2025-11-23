import argparse

from seqrec.runner import Runner
from seqrec.utils import parse_command_line_args
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SASRec', help='Model name, options: SASRec, GRU4Rec')
    parser.add_argument('--dataset', type=str, default='Games_5core', help='Source domain')
    parser.add_argument('--exp_type', type=str, default='srec')
    parser.add_argument('--embedding', type=str, default='./item_info/Games_5core/LLM2Vec_Qwen2-0.5B-Backbone_title_item_embs.npy', help='Whether to use source domain data')
    parser.add_argument('--seq_embedding', type=str, default='', help='whether pre-trained sequence embeddings are used')

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    args_dict = vars(args)  

    merged_dict = {**args_dict, **command_line_configs}


    runner = Runner(
        model_name=args.model,
        config_dict= merged_dict
    )
    runner.run()


# CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port=12324 main.py --model=PDSRec --sd=T --td=T 

