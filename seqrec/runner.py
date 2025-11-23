import torch
import numpy as np
from typing import Union

from accelerate import Accelerator
from torch.utils.data import DataLoader, Sampler
import wandb

from .recdata import NormalRecData
from .base import AbstractModel

from .utils import get_config, init_device, init_seed, get_model, get_file_name, diagonalize_and_scale
from .trainer import BaseTrainer


class Runner:
    def __init__(
            self,
            model_name: Union[str, AbstractModel],
            config_dict: dict = None,
            config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            config_file=config_file,
            config_dict=config_dict
        )
        print(self.config)

        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device()

        wandb.init(
            project="LLM2Rec_Eval",  # Replace with your project name
            name=get_file_name(self.config),              # Set the desired run name
        )
        self.accelerator = Accelerator(log_with='wandb')

        self.config['accelerator'] = self.accelerator

        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        _ = NormalRecData(self.config).load_data()

        self.recdata = {
            'train': _[0],
            'valid': _[1],
            'test': _[2]
        }
        self.config['select_pool'] = _[3]
        self.config['item_num'] = _[4]
        self.config['eos_token'] = _[4] + 1

        if self.config['embedding']:
            pretrained_item_embeddings = torch.tensor(np.load(self.config['embedding']), dtype=torch.float32).to(self.config['device'])
            # judge if "seq_embedding" in config.keys()
            if "seq_embedding" in self.config.keys() and self.config['seq_embedding']:
                base_seq_embedding_path = self.config['seq_embedding']
                train_seq_embedding_path = base_seq_embedding_path.format("train")
                valid_seq_embedding_path = base_seq_embedding_path.format("val")
                test_seq_embedding_path = base_seq_embedding_path.format("test")
                train_seq_embedding = torch.tensor(np.load(train_seq_embedding_path), dtype=torch.float32).to(self.config['device'])
                valid_seq_embedding = torch.tensor(np.load(valid_seq_embedding_path), dtype=torch.float32).to(self.config['device'])
                test_seq_embedding = torch.tensor(np.load(test_seq_embedding_path), dtype=torch.float32).to(self.config['device'])
                pretrained_item_embeddings = [pretrained_item_embeddings, train_seq_embedding, valid_seq_embedding, test_seq_embedding]

        else:
            pretrained_item_embeddings = None

        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, pretrained_item_embeddings)

        print(self.model)
        # print(self.model.n_parameters)
        self.trainer = BaseTrainer(self.config, self.model)

    def run(self):
        train_dataloader = DataLoader(
                self.recdata['train'],
                batch_size=self.config['train_batch_size'],
                shuffle=True,
            )
        val_dataloader = DataLoader(
            self.recdata['valid'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
        )
        test_dataloader = DataLoader(
            self.recdata['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
        )

        # skip training for ItemKNN model
        if self.config['model'] != 'ItemKNN':
            self.trainer.train(train_dataloader, val_dataloader)

            self.accelerator.wait_for_everyone()
            self.model = self.accelerator.unwrap_model(self.model)

            if self.config.get('steps', None) != 0:
                self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))
            else:
                """
                SASRec: ckpt/PDSRec-main.py_--model=SASRec_--sd=B_--td=B_--loss_type=ce_--lr=1e-2_--exp_type=lr-Sep-14-2024_09-29-11-ac20ba.pth
                DreamRec: ckpt/PDSRec-main.py_--model=DreamRec_--sd=B_--td=B_--hidden_size=3072_--exp_type=dim-Sep-13-2024_15-26-48-9c76db.pth
                Ours: ckpt/PDSRec-main.py_--model=PDSRec_--sd=B_--td=B_--loss_type=cosine_--ab=iids_--hidden_size=3072_--exp_type=ab-Sep-14-2024_15-53-12-06c9df.pth
                """
                ckpt_dict = {
                    'SASRec': 'ckpt/PDSRec-main.py_--model=SASRec_--sd=B_--td=B_--loss_type=ce_--lr=1e-2_--exp_type=lr-Sep-14-2024_09-29-11-ac20ba.pth',
                    'DreamRec': 'ckpt/PDSRec-main.py_--model=DreamRec_--sd=B_--td=B_--hidden_size=3072_--exp_type=dim-Sep-13-2024_15-26-48-9c76db.pth',
                    'PDSRec': 'ckpt/PDSRec-main.py_--model=PDSRec_--sd=B_--td=B_--loss_type=cosine_--ab=iids_--hidden_size=3072_--exp_type=ab-Sep-14-2024_15-53-12-06c9df.pth'
                }
                self.model.load_state_dict(torch.load(ckpt_dict[self.config['model']]))
                embeddings = self.model.get_current_embeddings()
                embeddings_np = embeddings.cpu().numpy()
                import numpy as np
                np.save('{}_{}_embeddings.npy'.format(self.config['model'], self.config['dataset']), embeddings_np)

            self.model, test_dataloader = self.accelerator.prepare(
                self.model, test_dataloader
            )
            if self.accelerator.is_main_process:
                print(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        if self.config.get('step', None) != 0:
            test_results = self.trainer.evaluate(test_dataloader)
            print(test_results)
            if self.accelerator.is_main_process:
                for key in test_results:
                    self.accelerator.log({f'Test_Metric/{key}': test_results[key]})

        if self.config['exp_type'] == 'check':
            np.save('{}_{}_vis_embeddings.npy'.format(self.config['model'], self.config['dataset']),
                    np.array(self.model.samples))
            np.save('{}_{}_pred_embeddings.npy'.format(self.config['model'], self.config['dataset']),
                    np.array(self.model.predict_embeddings.detach().cpu().numpy()))
            np.save('{}_{}_target_embeddings.npy'.format(self.config['model'], self.config['dataset']),
                    np.array(self.model.target_embedding.detach().cpu().numpy()))
        
        if self.accelerator.is_main_process:
            if self.config['save'] is False:
                import os
                if os.path.exists(self.trainer.saved_model_ckpt):
                    os.remove(self.trainer.saved_model_ckpt)
                    print(f"{self.trainer.saved_model_ckpt} has been deleted.")
                else:
                    print(f"{self.trainer.saved_model_ckpt} not found.")

        self.trainer.end()
        return test_results, self.config

