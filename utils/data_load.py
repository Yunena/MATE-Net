import torch
import numpy as np
from torch.utils.data import Dataset
import copy

from models.dimension_ascend import Ascender
from models.mate_net import AttentionModule
from .data_augmentation import *
import random
import os
import time
from tqdm import tqdm

from .model_load import load_model
from .resample import re_sample



class MATENetDataset(Dataset):
    def __init__(self,model_type,image_idx_list,image_path,mask_path,minmax_clinical_data,clinical_num, device, attn_model_path = None,zscored_clinical_data=None,pred_result_data=None):
        assert model_type in ['attn', 'pred']
        self.aim_shape = (8,32,32)
        self.model_type = model_type
        self.clinical_num = clinical_num
        self.image_idx_list = image_idx_list
        self.image_path = image_path
        self.mask_path = mask_path
        self.minmax_clinical_data = minmax_clinical_data
        self.zscore_clinical_data = zscored_clinical_data
        self.pred_result_data = pred_result_data
        self.asced_list = None
        self.image_data = self._image_load()
        if self.model_type == 'pred' and clinical_num>0:
            self.mask_data = self._mask_load()
            self.model_list = self._model_load(attn_model_path,device,clinical_num)
            self.asced_list = self._expanse_clinical(clinical_num,device)
            print(len(self.asced_list),self.asced_list[0].shape)
    
    def __len__(self):
        return len(self.image_idx_list)

    def __getitem__(self, idx):

        if self.model_type=='pred':
            if self.clinical_num==0:
                sample = {
                "image" : re_sample(torch.Tensor(self.image_data[idx]),self.aim_shape)[0],
                "result" : torch.Tensor(self.pred_result_data[idx])
                }
            else:       
                sample = {
                    "image" : re_sample(torch.Tensor(self.image_data[idx]),self.aim_shape)[0],
                    "asced" : re_sample(torch.Tensor(self.asced_list[idx]),self.aim_shape)[0],
                    "result" : torch.Tensor(self.pred_result_data[idx])
                }

        else:
            sample = {
                "image" : torch.Tensor(self.image_data[idx]),
                "result" : torch.Tensor(self.minmax_clinical_data[idx])
            }
        return sample

    def _image_load(self):
        image_list = []
        for idx in tqdm(self.image_idx_list):
            image_list.append(np.load(os.path.join(self.image_path,str(idx)+'.npy')))
        return image_list
    
    def _mask_load(self):
        image_list = []
        for idx in tqdm(self.image_idx_list):
            image_list.append(np.load(os.path.join(self.mask_path,str(idx)+'.npy')))
        return image_list

    def _model_load(self,model_path,device,clinical_num,model=AttentionModule()):
        model_list = []
        for j in range(clinical_num):
            model_ = load_model(model,os.path.join(model_path,'attn_module'+'_'+str(j)+'.pkl'))
            model_ = model_.to(device)
            model_.eval()
            model_list.append(copy.deepcopy(model_))
        return model_list


    def _expanse_clinical(self,clinical_num,device):
        if clinical_num==0:
            return None
        asc = Ascender(layer='model.conv1')
        asc_list = []
        for i,img in enumerate(tqdm(self.image_data)):
            mask = re_sample(self.mask_data[i],(32,128,128),'arr').numpy()
            clinical_wise = []
            img = np.array([img])
            img = torch.Tensor(img)
            img = img.to(device)
            for j in range(clinical_num):
                model = self.model_list[j]
                model.eval()
                batch_wise = []
                cam_img = asc.get_grad_cam(model,img)
                cam_img = cam_img*mask
                cam_img = cam_img * self.zscore_clinical_data[i][j]
                batch_wise.append(cam_img)
                clinical_wise.append(np.concatenate(batch_wise,axis=0))
            asc_list.append(np.concatenate(clinical_wise,axis=1))
        return asc_list
