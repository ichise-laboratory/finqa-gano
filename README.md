# GANO for Financial Question Answering
This repository contains the code and data to reproduce the experimental results in our paper: Enhancing Financial Table and Text Question Answering with Tabular Graph and Numerical Reasoning. 

## Summary
We proposed a model developed from the baseline of a financial QA dataset named [TAT-QA](https://nextplusplus.github.io/TAT-QA/). The baseline model, [TagOp](https://github.com/NExTplusplus/tat-qa), consists of answer span (evidence) extraction and numerical reasoning modules. As our main contributions, we introduced two components to the model: a GNN-based evidence extraction module for tables and an improved numerical reasoning module.

## Installation
You can install directly from our exported conda environment or follow our installation steps:
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c huggingface transformers==4.11.3
conda install pandas==1.4.4
conda install pyg==2.0.4 -c pyg
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install allennlp==2.8.0
pip install pytorch-lightning==1.5.8
pip install setuptools==59.5.0
pip install wandb==0.12.21
```

## Project Structure
Please see our supplemental material for data preparation and correction process. The processed dataset is in `data/finqa/tatqa/tagop/annotated`. We also provide reduced data for reproducing our experimental results in `data/finqa/tatqa/tagop/reduced`.

The code is in folder `gano`. The subfolder `gano/finqa/tatqa` contains the four models we included in our experiment (TagOp, NOC, GEE, and GANO). The other subfolder `gano/manage` contains experimental management script (e.g., naming and caching).

## Training and Testing
We included four models in our experiments: TagOp, NOC (TagOp + new number order classifier), GEE (GNN-based model), and GANO (NOC + GEE). Run the following commands to train the models:
```
python gano/finqa/tatqa/tagop/train.py bb --gpus 0, --cache --batch 16 
python gano/finqa/tatqa/noc/train.py bb --gpus 0, --cache --batch 16
python gano/finqa/tatqa/geer/train.py bb --gpus 0, --cache --batch 16 --gnn sage
python gano/finqa/tatqa/gano/train.py bb --gpus 0, --cache --batch 16 --gnn sage
```

The training script should generate logs in `logs/finqa/tatqa` and output in `outputs/finqa/tatqa` and save the model in `models/finqa/tatqa`. To test the model, locate the training output, e.g., `t.bb.b16.e5.1e-3`, then run the testing script (the default option uses the development set for testing):
```
python gano/finqa/tatqa/tagop/test.py t.bb.b16.e5.1e-3 --batch 16 --gpus 0,
```

If you want to submit the answers to the TAT-QA authors for evaluation, run the following command to generate answers from TAT-QA's test set:
```
python gano/finqa/tatqa/gano/predict.py t.bb.b16.e5.1e-3 --batch 16 --gpus 0,
```

## Citation
Please cite our work using the following citation:
```
@inproceedings{nararatwong-etal-2022-gano,
    title = "Enhancing Financial Table and Text Question Answering with Tabular Graph and Numerical Reasoning"}
    booktitle = "Proceedings of The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing"
    month = "November"
    year = "2022"
    address = "Online"
}
```

## Contact
Please see the paper for our contact information.