### 1. Initialize a model

from models.mate_net import *

attn_model = AttentionModule(
    input_num = 7                       # CT images quantity
)
pred_model = PredictionModule(
    clinical_num = 13,                  # Clinical data quantity
    input_num = 7                       # CT images quantity
)

### 2. Interprete a pred model

from models.mate_net import *
from wrapper import Model
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
if not os.path.exists('./results'):
    os.mkdir('./results')



model = Model(
    device='cuda:0',                    # Devices 
    save_path='results',                # Saving path
    model_type = 'pred',                 # Attention Module or Prediction Module
)

train_dataloader = model.get_dataloader(
    image_idx_list=list(range(20)),
    image_path = './example/train/image',
    mask_path = './example/train/mask',
    minmax_clinical_data= np.load('./example/train/minmax_clinical_data.npy'),
    clinical_num=13,
    device='cuda:0',
    attn_model_path='./results',
    zscored_clinical_data=np.load('./example/train/zscored_clinical_data.npy'),
    pred_result_data=np.load('./example/train/results.npy'),
    batch_size=20
)

validate_dataloader = model.get_dataloader(
    image_idx_list=list(range(5)),
    image_path = './example/validate/image',
    mask_path = './example/validate/mask',
    minmax_clinical_data= np.load('./example/validate/minmax_clinical_data.npy'),
    clinical_num=13,
    device='cuda:0',
    attn_model_path='./results',
    zscored_clinical_data=np.load('./example/validate/zscored_clinical_data.npy'),
    pred_result_data=np.load('./example/validate/results.npy'),
    batch_size=1
)

value1 = model.interprete(
    model=pred_model,
    base_loader=train_dataloader,
    data_loader=validate_dataloader
)

interprete_dataloader = model.get_dataloader(
    image_idx_list=list(range(20)),
    image_path = './example/train/image',
    mask_path = './example/train/mask',
    minmax_clinical_data= np.load('./example/train/minmax_clinical_data.npy'),
    clinical_num=13,
    device='cuda:0',
    attn_model_path='./results',
    zscored_clinical_data=np.load('./example/train/zscored_clinical_data.npy'),
    pred_result_data=np.load('./example/train/results.npy'),
    batch_size=1
)

value2 = model.interprete(
    model=pred_model,
    base_loader=train_dataloader,
    data_loader=interprete_dataloader
)

value = np.concatenate((value2,value1))
np.save('./results/shap_map.npy',value)

### 3 Generate c-shap map

from interpretability.cshap_generate import *
cshapmap = generate_cshap(generate_shaplist(value,[7,5,10]))    #7,10,5 is the selected 3 top index indicators
np.save('./results/cshap_map.npy',cshapmap)