# KG-LSC
This is the implementation for Knowledge Graph Embedding for Unsupervised Lexical Semantic Change Detection, the course project of CSC2611 at University of Toronto.

## Requirements
    python == 3.8.5
    torch == 1.8.1
    numpy == 1.19.2
    pandas == 1.0.1
    tqdm == 4.61.0
    tensorboardx == 2.5.1


## Run

- Pretraining Stage (The knowledge graphs will be automatically constructed along with the first run)
    > `python main.py --mode common`

- Finetuning Stage
    > `python main.py --mode respective`

- Evaluation
    > `python main.py --mode evaluate`

## Configurations

Tunable hyperparameters:
- bs: batchsize
- lr: learning rate
- emb_dim: embedding dimension
- base_model: DistMult or TransE
- num_ng: number of corrupted negative samples for each positive sample
- wd: weight decay
- scoring_fct_norm: N norm for TransE
- src: both (with NER) or only (without NER)
- epochs_common: maximum pretraining epochs
- epochs_respective: maximum finetuning epochs
- save_common_every: save interval for pretraining
- save_respective_every: save interval for finetuning

Auxilliary configurations:
- mode: pretraining, fintuning, or evaluate
- verbose: whether to print log on the screen
- num_workers: number of threads in for data loading
- gpu: an available gpu id
- seed: random seed
- save_root: your preferred directory to save data and models
