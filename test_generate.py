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
from train import test_generation

parser = argparse.ArgumentParser(description='test_generate.py')
parser.add_argument('-input_path', type=str, default='LiveBot', help="input folder path")
parser.add_argument('-workspace_path', type=str, default='LiveBot/MyPLVCG', help="output and config folders path")
parser.add_argument('-model_cfg_file', type=str, default=os.path.join('config', 'model.json'), help="model config file")
parser.add_argument('-generate_cfg_file', type=str, default=os.path.join('config', 'generate.json'), help="generate config file")
parser.add_argument('-img_file', type=str, default='res18.pkl', help="image file")
parser.add_argument('-corpus_file', type=str, default='train-context.json', help="train corpus json file")
parser.add_argument('-test_corpus_file', type=str, default='test-context.json', help="evaluation corpus json file")
parser.add_argument('-vocab_file', type=str, default='dicts-30000_tokenizer.json', help="vocabulary json file")
parser.add_argument('-video_type_map_file', type=str, default='video_type.json', help="video type json file")
parser.add_argument('-merges_file', type=str, default='merges.txt', help="merge tokens")
parser.add_argument('-preprocess_dir', type=str, default='preprocessed_data', help="path of preprocessed files")
parser.add_argument('-model_file', type=str, default='best-model.pt', help="Restoring model file")
parser.add_argument('-save_dir', type=str, default='ckpt_ft', help="checkpoint folder")
parser.add_argument('-generate_dir', type=str, default='generate', help="generate folder")
parser.add_argument('-log_dir', type=str, default='log', help="log folder")
     

         



class GenerateConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    predict_batch_size: int = 1
    total_steps: int = 0 # total number of steps to train
    weight_decay: float = 0.0
    max_output_length: int = 20
    min_output_length: int = 2
    print_steps: int = 1
    classification_thread : float=0.0
    num_beams : int = 5
    vocab_size : int = 30007
    candidate_num : int = 5
    no_repeat_ngram_size : int = 2
    repetition_penalty : int = 2
    do_sample : bool = True
    top_p : float = 1.0
    top_k : int = 50
    temperature : float = 1.0
    
    @classmethod
    def load_from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))



def generate():
    
    opt = parser.parse_args()




    generate_cfg = GenerateConfig.load_from_json(os.path.join(opt.workspace_path, opt.generate_cfg_file))
    model_cfg = ModelConfig.load_from_json(os.path.join(opt.workspace_path, opt.model_cfg_file))
    

    
    img_file = os.path.join(opt.input_path, opt.img_file)
    test_corpus_file = os.path.join(opt.input_path, opt.test_corpus_file)
    vocab_file = os.path.join(opt.input_path, opt.vocab_file)
    merges_file = os.path.join(opt.input_path, opt.merges_file)
    video_type_map_file = os.path.join(opt.input_path, opt.video_type_map_file)
    preprocess_dir = os.path.join(opt.workspace_path, opt.preprocess_dir)
    save_dir = os.path.join(opt.workspace_path, opt.save_dir)
    generate_dir = os.path.join(opt.workspace_path, opt.generate_dir)
    log_dir = os.path.join(opt.workspace_path, opt.log_dir)

    model_file = os.path.join(save_dir, opt.model_file)


    log_filename = "generate_log.txt"
    log_filename = os.path.join(log_dir,log_filename)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(opt.log_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(opt)
    
    
    tokenizer = MyBartTokenizer(vocab_file, merges_file)

    test_data = Dataset(vocab_file, test_corpus_file, img_file, video_type_map_file, preprocess_dir, model_cfg, generate_cfg, imgs=None, is_training=False, type = 'generate')
    test_data.load_gengrate_dataset(tokenizer)
    test_data.load_dataloader()


    
    model = MyPLVCG(model_cfg, test_data.video_type_weight, type="fine_tuning")
    model.load_state_dict(torch.load(model_file))
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()

    with(torch.no_grad()):
        generated = test_generation(generate_cfg, model, test_data)

    res_f = open(os.path.join(generate_dir, 'generated.txt'),"w", encoding='utf8')
    contexts = test_data.contexts
    ground_truth = test_data.comments
    
    for gen,gt,ct in zip(generated,ground_truth,contexts):
        
        ct_decode = test_data.decode(ct)
        end = ct_decode.find("<PAD>")
        if end != -1:
            ct_decode = ct_decode[:end]
        res_f.write("%s\n\nground_truth:\n"%(ct_decode))
        for g in gt:
            g_decode = test_data.decode(g)
            end = g_decode.find("<EOS>")
            if end != -1:
                g_decode = g_decode[:end]
            res_f.write("\t%s\n"%(g_decode))
        res_f.write("\ngenerated:\n")
        for s in gen:
            end = s.find("<EOS>")
            if end != -1:
                s = s[:end]
            res_f.write("\t%s\n"%(s))
        res_f.write("\n=============================\n\n")
    
    


if __name__ == '__main__':
    generate()