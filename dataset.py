# encoding=utf-8
# modified from https://github.com/shmsw25/bart-closed-book-qa/blob/master/data.py

import os
import json
import re
import string
import numpy as np
import copy
import random

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler



from transformers import RobertaTokenizer
from transformers.tokenization_utils import AddedToken

class MyBartTokenizer(RobertaTokenizer):
    # merges and vocab same as Roberta
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<BOS>",
        eos_token="<EOS>",
        sep_token="<&&&>",
        cls_token="<s>",
        unk_token="<UNK>",
        pad_token="<PAD>",
        mask_token="<MASK>",
        visual_token="<VISUAL>",
        add_prefix_space=False,
        **kwargs
    ):
        visual_token = AddedToken(visual_token, lstrip=False, rstrip=False) if isinstance(visual_token, str) else visual_token

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        
    def _tokenize(self,text):
        return(text.split())

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = " ".join(tokens)
        return text

class Dataset(object):

    def __init__(self, vocab_file, corpus_file, img_file, video_type_map_file, preprocess_path, model_cfg, train_cfg, imgs=None, is_training=True, type = 'pretrain'):


        self.vocab = self.load_vocab(vocab_file)
        self.vocab_size = len(self.vocab)
        self.corpus = self.load_from_json(open(corpus_file, 'r', encoding='utf8'))
        self.video_type_map, self.video_type_weight = self.load_video_type_map(video_type_map_file)


        data_type = "train" if is_training else "dev"
        if type == "generate" :
            data_type = "generate"
        if type == "test":
            data_type = "test"
        print(data_type)
        self.preprocessed_file = os.path.join(preprocess_path,data_type)

        print("corpus length:",len(self.corpus))
        if imgs is None:
            self.imgs = torch.load(open(img_file, 'rb'))
        else:
            self.imgs = imgs

        
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.is_training = is_training
        self.type = type


    def __len__(self):
        return len(self.corpus)

    def load_from_json(self,corpus_file):
        corpus = []
        for line in corpus_file:
            corpus.append(json.loads(line))
        return corpus

    def load_video_type_map(self, video_type_map_file):
        with open(video_type_map_file, "r") as f:
            video_type_map,video_type_weight = json.load(f)
        return (video_type_map, video_type_weight)
    
    def load_vocab(self,vocab_file):
        vocab = json.load(open(vocab_file, 'r', encoding='utf8'))
        return vocab

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]



    def load_dataset(self,tokenizer):
        self.tokenizer = tokenizer

        

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types = json.load(f)
        else:
            print ("Start tokenizing...")
            video_ids = []
            video_times = []
            video_types = []
            contexts = []
            comments = []
            for data in self.corpus:#[:500]:
                if len(video_ids) % 100 == 0:
                    print(len(video_ids))
                video_ids.append(int(data['vid']))
                video_times.append(data['time']-1)
                video_types.append(0)
                contexts.append("<VISUAL> " + data['context'])
                if self.is_training:
                    comments.append("<BOS> " + data['target'] + " <EOS>")
                else:
                    comments.append("<BOS> " + data['target'][0] + " <EOS>")
            
            print(contexts[0])
            print(comments[0])
                        
            context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            comment_input = tokenizer.batch_encode_plus(comments, pad_to_max_length=True, max_length=self.model_cfg.max_comment_len, truncation=True, add_special_tokens=False)
            
            input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = comment_input["input_ids"], comment_input["attention_mask"]
            
            for j in range(len(decoder_input_ids)):
                if decoder_input_ids[j][-1] != 2 and decoder_input_ids[j][-1] != 0:
                    decoder_input_ids[j][-1] = 2
            
            preprocessed_data = [input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.contexts = input_ids
        self.comments = decoder_input_ids
        self.video_types = video_types
        self.dataset = MyLBDataset(self.model_cfg, self.train_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types,
                                   self.imgs, self.is_training, type=self.type)
        return self.dataset

    def load_gengrate_dataset(self, tokenizer):
        self.tokenizer = tokenizer

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types = json.load(f)
        else:
            print ("Start tokenizing...")
            video_ids = []
            video_times = []
            video_types = []
            contexts = []
            decoder_input_ids = []
            decoder_attention_mask = []
            for data in self.corpus:#[:500]:
                if len(video_ids) % 100 == 0:
                    print(len(video_ids))
                video_ids.append(int(data['vid']))
                video_times.append(data['time']-1)
                video_types.append(0)
                contexts.append("<VISUAL> " + data['target'])

                candidate = ["<BOS> " + c + " <EOS>" for c in data['comment']]
                candidates_input = tokenizer.batch_encode_plus(candidate, pad_to_max_length=True, max_length=self.model_cfg.max_comment_len, truncation=True, add_special_tokens=False)
                candidate_input_ids, candidate_attention_mask = candidates_input["input_ids"], candidates_input["attention_mask"]            
                decoder_input_ids.append(candidate_input_ids)
                decoder_attention_mask.append(candidate_attention_mask)
            
            print(contexts[0])
                        
            context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]

            for j in range(len(decoder_input_ids)):
                if decoder_input_ids[j][-1] != 2 and decoder_input_ids[j][-1] != 0:
                    decoder_input_ids[j][-1] = 2
            
         
            preprocessed_data = [input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.contexts = input_ids
        self.comments = decoder_input_ids
        self.dataset = MyLBDataset(self.model_cfg, self.train_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types,
                                   self.imgs, self.is_training, type=self.type)
        return self.dataset


    def load_test_dataset(self, tokenizer):
        self.tokenizer = tokenizer

        self.gts = []
        for data in self.corpus:
            self.gts.append(data["candidate"])
            
        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types, candidate_labels = json.load(f)
        else:
            print ("Start tokenizing...")
            video_ids = []
            video_times = []
            contexts = []
            decoder_input_ids = []
            decoder_attention_mask = []
            comment_labels = []
            video_types = []
            for data in self.corpus:#[:500]:
                if len(video_ids) % 100 == 0:
                    print(len(video_ids))
                video_ids.append(int(data['video']))
                video_times.append(data['time']-1)
                video_types.append(0)
                
                contexts.append("<VISUAL> " + data['context'])

                candidate = []
                candidate_labels = []
                for c in data["candidate"].keys():
                    candidate.append("<BOS> " + c + " <EOS>")
                    candidate_labels.append(data["candidate"][c])
                    
                    
                candidates_input = tokenizer.batch_encode_plus(candidate, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
                candidate_input_ids, candidate_attention_mask = candidates_input["input_ids"], candidates_input["attention_mask"]

                for j in range(len(candidate_input_ids)):
                    if candidate_input_ids[j][-1] != 2 and candidate_input_ids[j][-1] != 0:
                        candidate_input_ids[j][-1] = 2
                    
                decoder_input_ids.append(candidate_input_ids)
                decoder_attention_mask.append(candidate_attention_mask)
                comment_labels.append(candidate_labels)


                        
            context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            
            input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]
            
            for j in range(len(decoder_input_ids)):
                if decoder_input_ids[j][-1] != 2 and decoder_input_ids[j][-1] != 0:
                    decoder_input_ids[j][-1] = 2
            
            
            preprocessed_data = [input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types, candidate_labels]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.contexts = input_ids
        self.comments = decoder_input_ids
        self.dataset = MyLBTestDataset(self.model_cfg, self.train_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types,
                                   self.imgs, self.is_training, type=self.type)
        return self.dataset


    def load_classification_dataset(self,tokenizer):
        self.tokenizer = tokenizer

        if os.path.exists(self.preprocessed_file):
            print("Loading pre-tokenized data from {}".format(self.preprocessed_file))
            with open(self.preprocessed_file, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types = json.load(f)
        else:
            print ("Start tokenizing...")
            video_ids = []
            video_times = []
            contexts = []
            comments = []
            video_types = []
            for data in self.corpus:#[:500]:
                if len(video_ids) % 100 == 0:
                    print(len(video_ids))
                video_ids.append(int(data['vid']))
                video_times.append(data['time']-1)
                video_types.append(0)
                contexts.append("<VISUAL> " + data['context'])
                if self.is_training:
                    comments.append("<BOS> " + data['target'] + " <EOS>")
                else:
                    comments.append("<BOS> " + data['target'][0] + " <EOS>")
            
            print(contexts[0])
            print(comments[0])
                        
            context_input = tokenizer.batch_encode_plus(contexts, pad_to_max_length=True, max_length=self.model_cfg.max_context_len, truncation=True, add_special_tokens=False)
            comment_input = tokenizer.batch_encode_plus(comments, pad_to_max_length=True, max_length=self.model_cfg.max_comment_len, truncation=True, add_special_tokens=False)
            
            input_ids, attention_mask = context_input["input_ids"], context_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = comment_input["input_ids"], comment_input["attention_mask"]
            
            for j in range(len(decoder_input_ids)):
                if decoder_input_ids[j][-1] != 2 and decoder_input_ids[j][-1] != 0:
                    decoder_input_ids[j][-1] = 2
            
            preprocessed_data = [input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types]
            with open(self.preprocessed_file, "w") as f:
                json.dump(preprocessed_data, f)

        self.contexts = input_ids
        self.comments = decoder_input_ids
        self.dataset = MyLBClassificationDataset(self.model_cfg, self.train_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types,
                                   self.imgs, self.is_training, type=self.type)
        return self.dataset 
    

    def load_dataloader(self):
        self.dataloader = MyDataLoader(self.train_cfg, self.dataset, self.is_training)


    def evaluate(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        ems = []
        for (prediction, dp) in zip(predictions, self.corpus):
            ems.append(get_exact_match(prediction, dp['comment'].split()))
        return ems

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        prediction_dict = {dp["id"]:prediction for dp, prediction in zip(self.data, predictions)}
        save_path = os.path.join(self.args.output_dir, "{}predictions.json".format(self.args.prefix))
        with open(save_path, "w") as f:
            json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))

def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (prediction == groundtruth)



class MyLBDataset(Dataset):
    def __init__(self, model_cfg, train_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types, imgs, is_training= True, type="pretrain" ):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.is_training = is_training
        self.type = type
        self.video_ids = video_ids
        self.video_times = video_times
        self.video_types = video_types
        self.imgs = imgs
        
        self.device = torch.device("cuda")


        assert len(self.input_ids)==len(self.attention_mask)
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)

    def load_imgs(self,video_id,video_time):
        
        previous = [i for i in range(-self.model_cfg.max_n_clips + 1, 1)]

        V_t = torch.zeros([self.model_cfg.max_n_clips,self.model_cfg.dim], device = self.device)
        i = self.model_cfg.max_n_clips-1
        for t in previous[::-1]:
            if video_time + t >= 0 and video_time + t < len(self.imgs[video_id]):
                V_t[i] = torch.cat((self.imgs[video_id][video_time + t],self.imgs[video_id][video_time + t]), dim=-1)
                i -= 1
        return V_t.to(self.device)

        


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        visual = self.load_imgs(self.video_ids[idx], self.video_times[idx])
        video_type = self.video_types[idx]
        
        if self.is_training and self.type=="pretrain":
            train_input_id,train_attention_mask = self.infilling(self.input_ids[idx], self.attention_mask[idx])
        else:
            train_input_id,train_attention_mask = self.input_ids[idx], self.attention_mask[idx]
            
        return torch.LongTensor(train_input_id), torch.LongTensor(train_attention_mask), \
            self.decoder_input_ids[idx], self.decoder_attention_mask[idx], visual, video_type

    def infilling(self,input_id, attention_mask, mask_token_index=5, visual_token_index=6):
        train_input_id = copy.deepcopy(input_id)
        train_attention_mask = copy.deepcopy(attention_mask)
        masked_tokens = 0
        max_masked_tokens = self.train_cfg.max_masked
        
        j = 1
        for i,token in enumerate(input_id[1:]):
            prob = random.random()
            if prob < self.train_cfg.mask_prob:
                span_length = np.random.poisson(lam=self.train_cfg.poisson_lambda, size=1)[0]
                if masked_tokens >= max_masked_tokens:
                    break
                if masked_tokens + span_length > max_masked_tokens:
                    span_length = max_masked_tokens - masked_tokens       
                          
                if span_length == 0:
                    train_input_id.insert(j+1, mask_token_index)
                    train_attention_mask.insert(j+1, 1)
                    j += 2
                    masked_tokens += 1
                else:
                    train_input_id[j:j+span_length] = [mask_token_index]
                    train_attention_mask[j:j+span_length] = [1]

                    j += 1-span_length
                    masked_tokens += span_length
        
        if len(train_input_id) > len(input_id):
            train_input_id = train_input_id[0:len(input_id)]
            train_attention_mask = train_attention_mask[0:len(input_id)]
        elif len(train_input_id) < len(input_id):
            train_input_id += [0]*(len(input_id)-len(train_input_id))
            train_attention_mask += [0]*(len(attention_mask)-len(train_attention_mask))
        
        return(train_input_id,train_attention_mask)
                    


class MyLBTestDataset(MyLBDataset):
    def __init__(self, model_cfg, test_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types, imgs, is_training, type):
        super(MyLBTestDataset, self).__init__( model_cfg, test_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types, imgs, is_training, type )

    def __getitem__(self, idx):
    
        visual = self.load_imgs(self.video_ids[idx], self.video_times[idx])
        video_type = self.video_types[idx]
        
        candidate_len = len(self.decoder_input_ids[0])

        
        train_input_id = torch.LongTensor(self.input_ids[idx]).unsqueeze(dim=0)
        train_input_id_repeat = torch.repeat_interleave(train_input_id, repeats=candidate_len, dim=0)
        
        train_attention_mask =  torch.LongTensor(self.attention_mask[idx]).unsqueeze(dim=0)
        train_attention_mask_repeat = torch.repeat_interleave(train_attention_mask, repeats=candidate_len, dim=0)
        
        visual_reprat = torch.repeat_interleave(visual.unsqueeze(dim=0), repeats=candidate_len, dim=0)
        
        return train_input_id_repeat, train_attention_mask_repeat, \
            self.decoder_input_ids[idx], self.decoder_attention_mask[idx], visual_reprat, video_type


class MyLBClassificationDataset(MyLBDataset):

    def __len__(self):
        return len(self.input_ids) * self.cfg.negative_num
    
    def __init__(self, model_cfg, classification_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types, imgs, is_training, type):
        self.cfg = classification_cfg
        super(MyLBClassificationDataset, self).__init__( model_cfg, classification_cfg, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, video_ids, video_times, video_types, imgs, is_training, type )
        self.rids = []

    def __getitem__(self, idx):
        tid = idx // self.cfg.negative_num
        
        
        visual = self.load_imgs(self.video_ids[tid], self.video_times[tid])
        if (idx % self.cfg.negative_num) == 0:
            label = 1
            rid = tid
        else:    
            label = 0
            rid = random.randint(0,len(self.input_ids)-1)
            while self.video_ids[tid] == self.video_ids[rid]:
                rid = random.randint(0,len(self.input_ids)-1)
        
        self.rids.append(rid)
        return torch.LongTensor(self.input_ids[tid]), torch.LongTensor(self.attention_mask[tid]), \
            self.decoder_input_ids[rid], self.decoder_attention_mask[rid], visual, label 


class MyDataLoader(DataLoader):

    def __init__(self, train_cfg, dataset, is_training):
        if is_training:
            sampler=SequentialSampler(dataset)
            batch_size = train_cfg.batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = train_cfg.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)

