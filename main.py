### 1. Initialize a model

from models.mate_net import *

attn_model = AttentionModule(
    input_num = 7                       # CT images quantity
)
pred_model = PredictionModule(
    clinical_num = 13,                  # Clinical data quantity
    input_num = 7                       # CT images quantity
)




### 2. Train a model

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


### 3. Evaluate a model

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



### 4. Analyze model interpretability
# Here the method is for SHAP maps generation. 

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


### 5. C-SHAP Generation
# *interpretability.py* implements interpretability methods. 

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

