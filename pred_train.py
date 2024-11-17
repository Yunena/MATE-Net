### 1. Initialize a model

from models.mate_net import *

attn_model = AttentionModule(
    input_num = 7                       # CT images quantity
)
pred_model = PredictionModule(
    clinical_num = 13,                  # Clinical data quantity
    input_num = 7                       # CT images quantity
)

### 2. Train a pred model

from models.mate_net import *
from wrapper import Model
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
if not os.path.exists('./results'):
    os.mkdir('./results')

# Prediction Module Training

model = Model(
    device='cuda:0',                    # Devices 
    save_path='results',                # Saving path
    model_type = 'pred',                 # Attention Module or Prediction Module
    is_evaluate=True
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
    pred_result_data=np.load('./example/train/results.npy')
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
    pred_result_data=np.load('./example/validate/results.npy')
)


model.train(
    epoch = 10,                         # Max-epoch
    model = pred_model,                 # Attention Module or Prediction Module (corresponded to the model_type)
    data_loader=train_dataloader,
    eval_data_loader = validate_dataloader
)