# RLMRec: Representation Learning with Large Language Models for Recommendation

 The codes are associated with the following paper:

 >**Representation Learning with Large Language Models for Recommendation**  
 >Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang*


\* denotes corresponding author
<p align="center">
<img src="RLMRec.png" alt="RLMRec" />
</p>

In this paper, we propose a model-agnostic framework RLMRec that aims to enhance existing recommenders with LLM-empowered representation learning. It proposes a recommendation paradigm that integrates representation learning with LLMs to capture intricate semantic aspects of user behaviors and preferences. RLMRec incorporates auxiliary textual signals, develops a user/item profiling paradigm empowered by LLMs, and aligns the semantic space of LLMs with the representation space of collaborative relational signals through a cross-view alignment framework.

## Environment

The codes are written in Python 3.9.16 with the following dependencies.

- numpy == 1.24.3
- pytorch == 1.13.1 (GPU version)
- torch-scatter == 2.1.1
- torch-sparse == 0.6.17
- scipy == 1.10.1

##  Dataset

We utilized three public datasets to evaluate RLMRec:  *Amazon-book, Yelp,* and *Steam*. 

First of all, please unzip the data by running following commands.
 ```
 cd data/
 cat data.tar.gz0* > data.tar.gz
 tar zxvf data.tar.gz
 ```

Each dataset consists of a training set, a validation set, and a test set. During the training process, we utilize the validation set to determine when to stop the training in order to prevent overfitting.
```
- amazon(yelp/steam)
|--- trn_mat.pkl
|--- val_mat.pkl
|--- tst_mat.pkl
|--- usr_prf.pkl
|--- itm_prf.pkl
|--- usr_emb_np.pkl
|--- itm_emb_np.pkl
```

The `usr_prf.pkl` and `itm_prf.pkl` files store the generated profiles of users and items from ChatGPT. You can run the code
```python data/read_profile.py```
as an example to read the profiles.

The encoded semantic embeddings from the user/item profiles are stored in `usr_emb_np.pkl` and `itm_emb_np.pkl`.

## Examples to run the codes

The command to evaluate the backebone models and RLMRec is as follows. 

  - Backbone 

    ```python encoder/train_encoder.py --model {model_name} --dataset {dataset} --cuda 0```   

  - RLMRec-Con (Constrastive Alignment):

    ```python encoder/train_encoder.py --model {model_name}_plus --dataset {dataset} --cuda 0```

  - RLMRec-Gen (Generative Alignment):

    ```python encoder/train_encoder.py --model {model_name}_gene --dataset {dataset} --cuda 0```

Supported models/datasets:

* model_name:  `gccf`, `lightgcn`, `sgl`, `simgcl`, `dccf`, `autocf`
* dataset: `amazon`, `yelp`, `steam`

Hypeparameters:

* The hyperparameters of each model are stored in `encoder/config/modelconf` (obtained by grid-search).

 **For advanced usage of arguments, run the code with --help argument.**

**Thanks for your interest in our work**
