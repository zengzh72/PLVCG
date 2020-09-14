#encoding=utf-8

import torch
import torch.nn as nn
import os
import json
import argparse
import logging
import numpy as np

from typing import NamedTuple

from transformers import AdamW, get_linear_schedule_with_warmup


from dataset import MyBartTokenizer, Dataset
from models import Config as ModelConfig
from models import MyPLVCG
from train import train, inference, print_results

parser = argparse.ArgumentParser(description='fine_tune_generate.py')
parser.add_argument('-input_path', type=str, default='LiveBot', help="input folder path")
parser.add_argument('-workspace_path', type=str, default='LiveBot/MyPLVCG', help="output and config folders path")
parser.add_argument('-model_cfg_file', type=str, default=os.path.join('config', 'model.json'), help="model config file")
parser.add_argument('-fine_tuning_cfg_file', type=str, default=os.path.join('config', 'fine_tuning.json'), help="fine tuning config file")
parser.add_argument('-img_file', type=str, default='res18.pkl', help="image file")
parser.add_argument('-corpus_file', type=str, default='train-context.json', help="train corpus json file")
parser.add_argument('-eval_corpus_file', type=str, default='dev-context.json', help="evaluation corpus json file")
parser.add_argument('-vocab_file', type=str, default='dicts-30000_tokenizer.json', help="vocabulary json file")
parser.add_argument('-merges_file', type=str, default='merges.txt', help="merges file")
parser.add_argument('-video_type_map_file', type=str, default='video_type.json', help="video type json file")
parser.add_argument('-preprocess_dir', type=str, default='preprocessed_data', help="path of preprocessed files")
parser.add_argument('-pretrain_file', type=str, default=os.path.join('ckpt','best-model.pt'), help="pretrain model file")
parser.add_argument('-model_file', type=str, default=None, help="Restoring model file")
parser.add_argument('-save_dir', type=str, default='ckpt_ft', help="checkpoint folder")
parser.add_argument('-log_dir', type=str, default='log', help="log folder")
parser.add_argument('-eval', default=False, action='store_true', help="evaluate mod")
       

         



class FineTuningConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 8
    predict_batch_size: int = 8
    lr: int = 1e-5 # learning rate
    n_epochs: int = 9 # the number of epoch
    save_steps: int = 20 # interval for saving model
    print_steps: int = 1
    eval_steps: int = 10
    total_steps: int = 0 # total number of steps to train
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 1
    max_grad_norm : float = 1.0
    wait_step: int = 50000
    max_output_length: int = 20
    
    @classmethod
    def load_from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))



def fine_tuning():
    
    opt = parser.parse_args()
    fine_tuning_cfg = FineTuningConfig.load_from_json(os.path.join(opt.workspace_path, opt.fine_tuning_cfg_file))
    model_cfg = ModelConfig.load_from_json(os.path.join(opt.workspace_path, opt.model_cfg_file))
    img_file = os.path.join(opt.input_path, opt.img_file)
    corpus_file = os.path.join(opt.input_path, opt.corpus_file)
    eval_corpus_file = os.path.join(opt.input_path, opt.eval_corpus_file)
    vocab_file = os.path.join(opt.input_path, opt.vocab_file)
    merges_file = os.path.join(opt.input_path, opt.merges_file)
    video_type_map_file = os.path.join(opt.input_path, opt.video_type_map_file)
    preprocess_dir = os.path.join(opt.workspace_path, opt.preprocess_dir)
    if not os.path.exists(preprocess_dir):
        os.mkdir(preprocess_dir)
    save_dir = os.path.join(opt.workspace_path, opt.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    log_dir = os.path.join(opt.workspace_path, opt.log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if opt.model_file is not None:
        model_file = os.path.join(save_dir, opt.model_file)
    else:
        model_file = None
    pretrain_file = os.path.join(opt.workspace_path, opt.pretrain_file)
    
    log_filename = "{}log.txt".format("" if not opt.eval else "eval_")
    log_filename = os.path.join(log_dir,log_filename)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(opt.log_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(opt)
    
    
    tokenizer = MyBartTokenizer(vocab_file, merges_file)

    dev_data = Dataset(vocab_file, eval_corpus_file, img_file, video_type_map_file, preprocess_dir, model_cfg, fine_tuning_cfg, imgs=None, is_training=False, type = 'fine_tuning')
    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    if opt.eval is False:
        train_data = Dataset(vocab_file, corpus_file, img_file, video_type_map_file, preprocess_dir, model_cfg, fine_tuning_cfg, imgs=dev_data.imgs, is_training=True, type = 'fine_tuning')
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader()
    
    
    model = MyPLVCG(model_cfg, dev_data.video_type_weight, type="fine_tuning")
    model.load_state_dict(torch.load(pretrain_file))
    
    if opt.eval is False:
        # Train
        if model_file is not None:
            model.load_state_dict(torch.load(model_file))
        
        optimizer = AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=fine_tuning_cfg.lr, eps=fine_tuning_cfg.adam_epsilon)
        train(fine_tuning_cfg, logger, save_dir, model, train_data, dev_data, optimizer, type="fine_tuning")
    
    else:
        #Evaluation
        logger.info("Loading checkpoint from {}".format(model_file))
        model.load_state_dict(torch.load(model_file))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()


        with(torch.no_grad()):
            total_loss, predictions, predictions_type, logits = inference(fine_tuning_cfg, model, dev_data, type)


        print_results(save_dir, dev_data, 0, total_loss, predictions, predictions_type, dev_data.comments, dev_data.contexts, dev_data.video_types)

if __name__ == '__main__':
    fine_tuning()