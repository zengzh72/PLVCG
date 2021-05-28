# PLVCG Model
This is  the implementation of PLVCG model for Live Video Comment Generation based on [PyTorch](https://pytorch.org/) and [Huggingface's Transformers](https://github.com/huggingface/transformers).

This model is a seq-to-seq model to generate live video comments by their context comments and visual frames.

## Requirment
Install Transformers:
```
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```

## Dataset
You can download our dataset at [here](https://drive.google.com/drive/folders/1QEZzKEv0G52WE_z8_7f4QpIq1mcs7ea1) and put them into the `LiveBot` folder.

This dataset is based on [Livebot](https://arxiv.org/abs/1809.04938) and the raw data can be found at [livebot](https://github.com/lancopku/livebot).

## Config
All the parameters can be set at: `LiveBot\MyPLVCG\config`.

## Train
Praining step:
```
python pretrain.py 
```
Generate fine-tuning step:
```
python fine_tune_generate.py
```
Classificatin fine-tuning step:
```
python fine_tune_classification.py
```

## Test:
Candidate comments ranking:
```
python test_rank.py 
```
Generate comments:
```
python test_generate.py 
```

## Citation
If this is helpful for your work, please cite our paper as:
```
@inproceedings{DBLP:conf/pakdd/ZengGXT21,
  author    = {Zehua Zeng and
               Neng Gao and
               Cong Xue and
               Chenyang Tu},
  title     = {{PLVCG:} {A} Pretraining Based Model for Live Video Comment Generation},
  booktitle = {Advances in Knowledge Discovery and Data Mining - 25th Pacific-Asia
               Conference, {PAKDD} 2021, Virtual Event, May 11-14, 2021, Proceedings,
               Part {II}},
  year      = {2021}
}
```
