import argparse
import datetime
import numpy as np
from seqrec.runner import Runner
from seqrec.utils import parse_command_line_args
import os
import json
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SASRec_v2', help='Model name, options: SASRec, GRU4Rec')
    parser.add_argument('--dataset', type=str, default='Games_5core', help='Source domain')
    parser.add_argument('--exp_type', type=str, default='srec')

    parser.add_argument('--embedding', type=str, default='', help='Whether to use source domain data')
    parser.add_argument('--seq_embedding', type=str, default='', help='whether pre-trained sequence embeddings are used')

    # parser.add_argument('--embedding', type=str, default='./item_info/Baby_AF2021/LLM2Vec_Mix6_SeqAug_Mistral-7B_step600_title_item_embs.npy', help='Whether to use source domain data')
    # parser.add_argument('--seq_embedding', type=str, default='./item_info/Baby_AF2021/LLM2Vec_Mix6_SeqAug_Mistral-7B_step600_{}_seq_embs.npy', help='whether pre-trained sequence embeddings are used')

    return parser.parse_known_args()



def calculate_mean_and_std(results):
    metrics = {}
    for result in results:
        for key, value in result.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)
    
    stats = {}
    for metric, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        stats[metric] = (float(mean), float(std))
    
    return stats



if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    args_dict = vars(args)  # 将 args 转换为字典

    # 合并字典，假设 command_line_configs 是一个字典
    merged_dict = {**args_dict, **command_line_configs}


    exp_seeds = [2024, 2025, 2026]
    test_results = []
    for seed in exp_seeds:
        merged_dict['rand_seed'] = seed

        runner = Runner(
            model_name=args.model,
            config_dict= merged_dict
        )
        test_result, exp_config = runner.run()

        test_results.append(test_result)
    
    # calcuate average and std of test results
    stats = calculate_mean_and_std(test_results)

    result_save_dir = f"./Results/{exp_config['dataset']}/{exp_config['model']}/lr_{exp_config['lr']}_dr_{exp_config['dropout']}_time_{datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')}_emb_{exp_config['embedding']}"
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    # write the stats to local txt file
    with open(f'{result_save_dir}/results.txt', 'a') as f:
        f.write(f"Final Results for {exp_config['model']} on {exp_config['dataset']}:\n")
        for key, value in stats.items():
            f.write(f'{key}: {value}\n')
        
        # write the results of each experiment
        f.write("\n\n")
        f.write("Results of each experiment:\n")
        for i, result in enumerate(test_results):
            f.write(f"Experiment {i+1}:\n")
            for key, value in result.items():
                f.write(f'{key}: {value}\n')
            f.write("\n")
    
    # save config as pretty json file
    with open(f'{result_save_dir}/config.json', 'w') as f:
        json.dump(merged_dict, f, indent=4)

    print("Finished:")
    print(stats)

# CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port=12324 main.py --model=PDSRec --sd=T --td=T 

