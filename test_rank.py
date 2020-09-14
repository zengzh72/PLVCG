#encoding=utf-8

import torch

import os
import json
import argparse
import logging
import random
import numpy as np

from typing import NamedTuple



from dataset import MyBartTokenizer, Dataset
from models import Config as ModelConfig
from models import MyPLVCG, MyClassificationPLVCG
from train import test_rank

parser = argparse.ArgumentParser(description='test_rank.py')
parser.add_argument('-input_path', type=str, default='LiveBot', help="input folder path")
parser.add_argument('-workspace_path', type=str, default='LiveBot/MyPLVCG', help="output and config folders path")
parser.add_argument('-model_cfg_file', type=str, default=os.path.join('config', 'model.json'), help="model config file")
parser.add_argument('-rank_cfg_file', type=str, default=os.path.join('config', 'rank.json'), help="pretrain config file")
parser.add_argument('-img_file', type=str, default='res18.pkl', help="image file")
parser.add_argument('-test_corpus_file', type=str, default='test-candidate.json', help="test corpus json file")
parser.add_argument('-vocab_file', type=str, default='dicts-30000_tokenizer.json', help="vocabulary json file")
parser.add_argument('-merges_file', type=str, default='merges.txt', help="merge tokens")
parser.add_argument('-video_type_map_file', type=str, default='video_type.json', help="video type json file")
parser.add_argument('-preprocess_dir', type=str, default='preprocessed_data', help="path of preprocessed files")
parser.add_argument('-save_dir', type=str, default='ckpt_cf', help="checkpoint folder")
parser.add_argument('-model_file', type=str, default='best-model.pt', help="Restoring model file")
parser.add_argument('-rank_dir', type=str, default='rank', help="rank folder")
parser.add_argument('-log_dir', type=str, default='log', help="log folder")
parser.add_argument('-load', default=False, action='store_true', help="load scores")     
parser.add_argument('-model_from', type=str, default='classification', help="the type of model to load")   
         



class RankConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    predict_batch_size: int = 1
    total_steps: int = 0 # total number of steps to train
    weight_decay: float = 0.0
    max_output_length: int = 20
    print_steps: int = 100
    classification_thread : float=0.0
    
    
    @classmethod
    def load_from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))



def ranking():
    
    opt = parser.parse_args()


    rank_config = RankConfig.load_from_json(os.path.join(opt.workspace_path, opt.rank_cfg_file))
    model_cfg = ModelConfig.load_from_json(os.path.join(opt.workspace_path, opt.model_cfg_file))
    

    
    img_file = os.path.join(opt.input_path, opt.img_file)
    test_corpus_file = os.path.join(opt.input_path, opt.test_corpus_file)
    vocab_file = os.path.join(opt.input_path, opt.vocab_file)
    merges_file = os.path.join(opt.input_path, opt.merges_file)
    video_type_map_file = os.path.join(opt.input_path, opt.video_type_map_file)
    preprocess_dir = os.path.join(opt.workspace_path, opt.preprocess_dir)
    rank_dir = os.path.join(opt.workspace_path, opt.rank_dir)
    log_dir = os.path.join(opt.workspace_path, opt.log_dir)
    save_dir = os.path.join(opt.workspace_path, opt.save_dir)
    model_file = os.path.join(save_dir, opt.model_file)
    

    log_filename = "{}log.txt".format("rank_")
    log_filename = os.path.join(log_dir,log_filename)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(opt.log_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(opt)
    
    
    tokenizer = MyBartTokenizer(vocab_file, merges_file)


    test_data = Dataset(vocab_file, test_corpus_file, img_file, video_type_map_file, preprocess_dir, model_cfg, rank_config, imgs=None, is_training=False, type = 'test')
    test_data.load_test_dataset(tokenizer)
    test_data.load_dataloader()
    
    
    if opt.model_from == 'classification':
        model = MyClassificationPLVCG(model_cfg, type='test')
        logger.info("Loading checkpoint from {}".format(model_file))
        model.load_state_dict(torch.load(model_file))
    else:
        model = MyPLVCG(model_cfg, type='test')
        logger.info("Loading checkpoint from {}".format(model_file))
        model.load_state_dict(torch.load(model_file))
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))
    model.eval()

    if opt.load:
        with open(os.path.join(rank_dir,'rank_score_%s.json'%(opt.model_from)), "r") as f:
            scores, pred_list = json.load(f)
            ranks = [sorted(range(len(score)), key=lambda k: score[k],reverse=True) for score in scores]
            # ============================= for random ================================
            #``random.shuffle (ranks )
            # ============================= for random ================================
    else:
        with(torch.no_grad()):
            ranks, scores, pred_list = test_rank(rank_config, model, test_data, type='classification')
        f_scores =  open(os.path.join(rank_dir,'rank_score_%s.json'%(opt.model_from)),'w', encoding='utf8')
        scores = [np.array(s.cpu()).tolist() for s in scores]
        json.dump([scores,pred_list], f_scores)
    
    predictions = []
    references = []
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    mean_rank = 0
    mean_reciprocal_rank = 0
    
    f_outs =  open(os.path.join(rank_dir,'out.txt'),'w', encoding='utf8')
    
    for i, rank in enumerate(ranks):
        gt_dic = test_data.gts[i]
        pred_b = pred_list[i]


        candidate = []
        comments = list(gt_dic.keys())
        for id in rank:
            candidate.append(comments[id])
        f_outs.write("\n========================\n")
        predictions.append(candidate)
        references.append(gt_dic)
        
        hit_rank = calc_hit_rank(candidate, gt_dic)
        
        f_outs.write("%d\n"%(hit_rank))
        cont = test_data.decode(test_data.contexts[i])
        end = cont.find("<PAD>")
        if end != -1:
            cont = cont[:end]


        f_outs.write("%s\n"%(cont))
        for j,id in enumerate(rank):
            
            
            if opt.model_from == 'classification':
                p = pred_b
                f_outs.write("%d %d %d %f %d %s || %d\n"%(i,j,rank[j],scores[i][rank[j]],gt_dic[comments[id]],comments[id],p))
            else:
                p = pred_b[rank[j]]
                f_outs.write("%d %d %d %f %d %s || %s\n"%(i,j,rank[j],scores[i][rank[j]],gt_dic[comments[id]],comments[id],p))
        

        mean_rank += hit_rank
        mean_reciprocal_rank += 1.0/hit_rank
        hits_1 += int(hit_rank <= 1)
        hits_5 += int(hit_rank <= 5)
        hits_10 += int(hit_rank <= 10)

        #for j,g in enumerate(gt_dic.keys()):
        #    print(scores[i][j], g, gt_dic[g])
    f_outs.close()
    total = len(test_data.gts)
    
    f_o = open(os.path.join(rank_dir, 'rank_res.txt'),'w', encoding='utf8')
    print("\t r@1:%f \t r@5:%f \t r@10:%f \t mr:%f \t mrr:%f"%(hits_1/total*100,hits_5/total*100,hits_10/total*100,mean_rank/total,mean_reciprocal_rank/total))
    f_o.write("\t r@1:%f \t r@5:%f \t r@10:%f \t mr:%f \t mrr:%f"%(hits_1/total*100,hits_5/total*100,hits_10/total*100,mean_rank/total,mean_reciprocal_rank/total))
    
        
        
        
def calc_hit_rank(prediction, reference):
    for i, p in enumerate(prediction):
        if reference[p] == 1:
            #print(i,p,reference[p])
            return i+1
    print(prediction)
    print(reference)
    raise ValueError('No reference!')

def recall(predictions, references, k=1):
    assert len(predictions) == len(references)
    total = len(references)
    hits = 0
    for p, c in zip(predictions, references):
        hits += int(calc_hit_rank(p, c) <= k)
    return hits * 100.0 / total   

if __name__ == '__main__':
    ranking()