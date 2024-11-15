# MATE-Net

## Introduction
This repository includes the implementation of Multimodal and ATtention-based Expansion Network (MATE-Net) for acute ischemic stroke (AIS) patients in 90 days mRS prediction. 

MATE-Net performed favorable prognostic performance and possessed interpretability. We proposed a biomarker, i.e.,  Combined SHAP (C-SHAP) to help clinical prognosis assessment. 

## Prerequisites
The implementation was based on Python 3.8 and the following dependencies: 
1. matplotlib==3.5.1 
2. medcam==0.1.21 
3. nibabel==3.2.2 
4. numpy==1.22.3 
5. pandas==1.4.3 
6. scikit_learn==1.0.2 
7. shap==0.39.0 
8. torch==1.13.1 
9. torchio==0.18.30 
10. torchmetrics==0.11.4 
11. tqdm==4.65.0 

## Documentation
*wrapper.py* contains the interfaces for training, evaluation and interpretability analysis. 

#### *main.py* covers the code below. 
### 1. Initialize a model
```python
from models.mate_net import *

attn_model = AttentionModule(
    input_num = 7                       # CT images quantity
)
pred_model = PredictionModule(
    clinical_num = 13,                  # Clinical data quantity
    input_num = 7                       # CT images quantity
)

```


### 2. Train a model
```python
from models.mate_net import *
from wrapper import Model

model = Model(
    jsonfile='jsonfiles/jsonfile',      # dataset and general configuration for the experiment
    devices=0                           # GPU device id
)



model.kfold_train(                      # choose k-fold stratege
    attn_model=attn_model,              # AttentionModule
    pred_model=pred_model,              # PredictionModule
    epoch=80,                           # training epoch
    value_path='results',               # results saving path (model path)
    start_idx=0,                        # the start saving path for the following k-fold
    training_type='multimodal',         # model type
    kvalue=5                            # 5-fold
)

```
### 3. Evaluate a model
```python
from models.mate_net import *
from wrapper import Model

model = Model(
    jsonfile='jsonfiles/jsonfile',      # dataset and general configuration for the experiment
    devices=0                           # GPU device id
)

model.evaluate(
    attn_model=attn_model,              # AttentionModule
    pred_model=pred_model,              # PredictionModule
    value_path='results',               # results saving path (model path)
    training_type='multimodal',         # model type
    start_idx=0,                        # the start saving path for the following k-fold
    result_path='bmultimodaleval.txt',  # saving path
    kvalue=1                            # number of models
)

```

### 4. Analyze model interpretability
Here the method is for SHAP maps generation. 
```python
from models.mate_net import *
from wrapper import Model

model = Model(
    jsonfile='jsonfiles/jsonfile',      # dataset and general configuration for the experiment
    devices=0                           # GPU device id
)

model.interprete(
    attn_model=attn_model,              # AttentionModule
    pred_model=pred_model,              # PredictionModule
    value_path='results',               # results saving path (model path)
    start_idx=0,                        # the start saving path for the following k-fold
    training_type='multimodal',         # model type
    kvalue=1                            # number of models
)
```

### 5. C-SHAP Generation
*interpretability.py* implements interpretability methods. 
```python
from interpreter import Interpreting

inter = Interpreting(
    attn_model=attn_model,              # AttentionModule
    value_path='results',               # results saving path (model path)
    idx=0,                              # the start saving path for the following k-fold
    jsonfile='jsonfiles/jsonfile',      # dataset and general configuration for the experiment
    devices=0,                          # GPU device id
    inte_type='cshap',                  # interpretability type
)

inter.get_cshap(
    name1='multimodal_values0.npy',     # shap map saving path
    save_path='cshap.npy',              # generated cshap saving path
    )
```

## DEMO
The C-SHAP values for the four individuals in Figure. 5 in the text are presented in the *demo* folder.

