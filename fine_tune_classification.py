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
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import MyBartTokenizer, Dataset
from models import Config as ModelConfig
from models import MyClassificationPLVCG
from train import train, inference, print_classification_res

parser = argparse.ArgumentParser(description='fine_tune_classification.py')
parser.add_argument('-input_path', type=str, default='LiveBot', help="input folder path")
parser.add_argument('-workspace_path', type=str, default='LiveBot/MyPLVCG', help="output and config folders path")
parser.add_argument('-model_cfg_file', type=str, default=os.path.join('config', 'model.json'), help="model config file")
parser.add_argument('-classification_cfg_file', type=str, default=os.path.join('config', 'classification.json'), help="pretrain config file")
parser.add_argument('-img_file', type=str, default='res18.pkl', help="image file")
parser.add_argument('-corpus_file', type=str, default='train-context.json', help="train corpus json file")
parser.add_argument('-eval_corpus_file', type=str, default='dev-context.json', help="evaluation corpus json file")
parser.add_argument('-vocab_file', type=str, default='dicts-30000_tokenizer.json', help="vocabulary json file")
parser.add_argument('-merges_file', type=str, default='merges.txt', help="merges file")
parser.add_argument('-preprocess_dir', type=str, default='preprocessed_data', help="path of preprocessed files")
parser.add_argument('-pretrain_file', type=str, default=os.path.join('ckpt_ft','best-model.pt'), help="pretrain model file")
parser.add_argument('-video_type_map_file', type=str, default='video_type.json', help="video type json file")
parser.add_argument('-model_file', type=str, default=None, help="Restoring model file")
parser.add_argument('-save_dir', type=str, default='ckpt_cf', help="checkpoint folder")
parser.add_argument('-log_dir', type=str, default='log', help="log folder")
parser.add_argument('-eval', default=False, action='store_true', help="evaluate mod")
parser.add_argument('-without_pretrain', default=False, action='store_true', help=" train without pretrain file")

         



class ClassificationConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 2
    predict_batch_size: int = 8
    lr: int = 1e-5 # learning rate
    n_epochs: int = 1 # the number of epoch
    warmup: float = 0.01
    warmup_steps: int = 0
    save_steps: int = 200 # interval for saving model
    print_steps: int = 100
    eval_steps: int = 1000
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    gradient_accumulation_steps: int = 1
    max_grad_norm : float = 1.0
    wait_step: int = 50000
    negative_num : int = 5
    classification_thread : float=0.6
    
    @classmethod
    def load_from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))



def classification():
    
    opt = parser.parse_args()




    classification_cfg = ClassificationConfig.load_from_json(os.path.join(opt.workspace_path, opt.classification_cfg_file))
    model_cfg = ModelConfig.load_from_json(os.path.join(opt.workspace_path, opt.model_cfg_file))
    

    
    img_file = os.path.join(opt.input_path, opt.img_file)
    corpus_file = os.path.join(opt.input_path, opt.corpus_file)
    eval_corpus_file = os.path.join(opt.input_path, opt.eval_corpus_file)
    vocab_file = os.path.join(opt.input_path, opt.vocab_file)
    merges_file = os.path.join(opt.input_path, opt.merges_file)
    preprocess_dir = os.path.join(opt.workspace_path, opt.preprocess_dir)
    video_type_map_file = os.path.join(opt.input_path, opt.video_type_map_file)
    save_dir = os.path.join(opt.workspace_path, opt.save_dir)
    log_dir = os.path.join(opt.workspace_path, opt.log_dir)
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

    dev_data = Dataset(vocab_file, eval_corpus_file, img_file, video_type_map_file, preprocess_dir, model_cfg, classification_cfg, imgs=None, is_training=False, type ='fine_tuning')
    dev_data.load_classification_dataset(tokenizer)
    dev_data.load_dataloader()

    if opt.eval is False:
        train_data = Dataset(vocab_file, corpus_file, img_file, video_type_map_file, preprocess_dir, model_cfg, classification_cfg, imgs=dev_data.imgs, is_training=True, type ='fine_tuning')
        train_data.load_classification_dataset(tokenizer)
        train_data.load_dataloader()
    
    
    model = MyClassificationPLVCG(model_cfg, classification_cfg.negative_num,  type="classification")

    if not opt.without_pretrain:
        model_dict =  model.state_dict()
        print("Loading pretrain file...")
        pretrained_dict = torch.load(pretrain_file)
    
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
        model_dict.update(pretrained_dict)
    
        model.load_state_dict(model_dict)
    
    gt = np.array(([1] + [0] * (5-1))*(100//5))
    
    if opt.eval is False:
        #Train
        if model_file is not None:
            model.load_state_dict(torch.load(model_file))
        

        optimizer = AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=classification_cfg.lr, eps=classification_cfg.adam_epsilon)
        train(classification_cfg, logger, save_dir, model, train_data, dev_data, optimizer, type = 'classification')
    
    else:
        #Evaluation
        checkpoint = os.path.join(save_dir, 'best-model.pt')
        model = MyClassificationPLVCG(model_cfg, classification_cfg.negative_num)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        model.load_state_dict(torch.load(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()

        with(torch.no_grad()):
            total_loss, predictions, logits = inference(classification_cfg, model, dev_data, 'classification')
        

        for i in range(10):
            print(0.3+i*0.05)
            predictions = logit_pred(logits,0.3+i*0.05)

            precision, recall ,f1 = metrics(predictions,classification_cfg.negative_num)
            print("precision:%f \t recall:%f \t f1:%f"%(precision, recall, f1))
            rids = dev_data.dataset.rids
            print_classification_res(save_dir, dev_data, 0, total_loss, predictions, dev_data.comments, dev_data.contexts, rids, logits, classification_cfg.negative_num)


def logit_pred(logits, thread = 0.5):
    softmax_logits = [torch.softmax(logit ,dim=-1) for logit in logits]
    pred = [logit[1].gt(thread).long() for logit in softmax_logits]
    return pred

def metrics(predictions,negative_num):
    pred = np.array([p.item() for p in predictions])
    gt = np.array(([1] + [0] * (negative_num-1))*(len(predictions)//negative_num))
    
    print(pred)
    print(gt)
    precision = precision_score(gt, pred)
    recall =  recall_score(gt, pred) 
    f1 = f1_score(gt, pred)
    
    return precision, recall ,f1
    

if __name__ == '__main__':
    classification()