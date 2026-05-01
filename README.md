# TCDA: Thread-Constrained Discourse-Aware Modeling for Conversational Sentiment Quadruple Analysis

This repository contains the source code and dataset for our paper.

##  Requirements

The model is implemented using PyTorch. We recommend using a virtual environment (Conda) for setup.

**Core Dependencies:**
+ python >= 3.7 (Tested on 3.7.9)
+ torch >= 1.8.1 (Tested on 1.9.0+cu111)

**Reproducible Environment:**
To strictly reproduce the scores reported in our paper, we used the following environment configuration:

| Package | Version |
| :--- | :--- |
| `python` | 3.7.9 |
| `torch` | 1.9.0+cu111 |
| `torchaudio` | 0.9.0 |
| `torchvision` | 0.10.0+cu111 |
| `transformers` | 4.20.1 |
| `numpy` | 1.21.6 |

*Note: If you use different versions, you may need to perform hyperparameter tuning to achieve optimal performance.*

**Installation:**
```bash
pip install -r requirements.txt
```


## TRAIN

```bash
# seed = 50-66
bash script/train.sh
```