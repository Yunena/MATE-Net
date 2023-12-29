import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import torchio as tio
import pandas as pd
import nibabel as nib
import operator
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

AUGMENTATION_NUM = 100
TARGET_SIZE = (256, 256, 32)
STEP = 4
NP_LIST = []
croporpadtrans = tio.CropOrPad(TARGET_SIZE)



class HybridDataset(Dataset):
    def __init__(self, idx_list, image_idx_list, image_path, norm_image_path, image_list, clinical_path, result_path,
                 attn_model,aug_time=1,mask_name='mask.nii',layer='model.conv1',pred=False,
                 model_path=None,clinical_num=13,include_list = None):
        # print('Init')
        print(image_path)
        self.include_list = include_list
        if self.include_list is None:
            self.include_list = list(range(clinical_num))
        self.ori_idx_list = idx_list
        self.ori_idx_list.sort()
        self.idx_list = idx_list * aug_time
        self.idx_list.sort()
        self.image_idx_list = pd.read_excel(image_idx_list, header=0)
        self.clinical_data = pd.read_excel(clinical_path, index_col=0, header=0)
        scaler = MinMaxScaler()
        self.model_list=[]
        self.normalized_clinical_data = pd.DataFrame(scaler.fit_transform(self.clinical_data),index = self.clinical_data.index)
        self.zscore_clinical_data = self.zscore_regression_value()
        self.result_data = pd.read_excel(result_path, index_col=0, header=0)
        self.image_list = image_list
        self.image_path = image_path
        self.norm_image_path = norm_image_path
        self.aug_time = aug_time
        self.asc = Ascender(layer=layer)
        self.mask_name = mask_name
        self.model_list=[]
        self.np_list,self.asc_list = self.all_augmentation(pred,model_path,clinical_num,attn_model)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        image_idx = self.image_idx_list.iloc[self.idx_list[idx], 0]
        clinical = self.clinical_data.loc[image_idx].iloc[self.include_list]
        result = self.result_data.loc[image_idx]
        normalized_clinical = self.normalized_clinical_data.loc[image_idx]

        if self.np_list is None:
            sample = {
                "image": torch.Tensor(np.load(self.temp_path + str(idx) + '.npy')), \
                "clinical": torch.Tensor(clinical), \
                "result": torch.Tensor(result),\
                "normalized":torch.Tensor(normalized_clinical)
            }
        else:
            if self.asc_list is None:
                sample = {"image": torch.Tensor(self.np_list[idx]), \
                          "clinical": torch.Tensor(clinical), \
                          "result": torch.Tensor(result),\
                          "normalized":torch.Tensor(normalized_clinical)
                }
            else:
                sample = {"image": torch.Tensor(self.np_list[idx]), \
                          "clinical": torch.Tensor(clinical), \
                          "result": torch.Tensor(result),\
                          "normalized":torch.Tensor(normalized_clinical),
                          "asced":torch.Tensor(self.asc_list[idx][self.include_list])
                }


        return sample

    def all_augmentation(self,pred=False,model_path=None,clinical_num=13,model = AttentionModule(7)):
        # print('Aug')
        nplist_list = []
        asclist_list = []
        pred = pred and clinical_num>0
        if pred:
            for j in tqdm(range(clinical_num)):
                model_ = load_model(model,os.path.join(model_path,'model'+str(j)+'.pkl'))
                model_ = model_.cuda()
                model_.eval()
                self.model_list.append(copy.deepcopy(model_))

        for idx in tqdm(self.ori_idx_list):
            # print(idx)
            image_idx = self.image_idx_list.iloc[idx, 0]
            image_name_tensor = []
            if pred:
                for i in range(self.aug_time):
                    # print('start',end=' ')
                    start = time.time()
                    if (image_name_tensor == []):
                        for image_name in self.image_list:
                            # print(image_name,end=' ')
                            image_file = os.path.join(self.image_path, str(image_idx), image_name)
                            scalarimage = tio.ScalarImage(image_file)
                            image_name_tensor.append(scalarimage)
                            del (scalarimage)
                        image_name_tensor.append(tio.ScalarImage(os.path.join(self.image_path, str(image_idx), self.mask_name)))
                    nplist = self.augornot(image_name_tensor, i)
                    mask = nplist[-1]
                    nplist = nplist[:-1]
                    asclist = self.generate_batch_gradcam(mask,model_path,torch.Tensor(np.array([nplist])).cuda(),
                                                          self.zscore_clinical_data.loc[image_idx])

                    nplist_list.append(nplist)
                    asclist_list.append(asclist[0])
                    del (nplist)
                    del asclist
            else:
                for i in range(self.aug_time):
                    start = time.time()
                    if (image_name_tensor == []):
                        for image_name in self.image_list:
                            image_file = os.path.join(self.image_path, str(image_idx), image_name)
                            scalarimage = tio.ScalarImage(image_file)
                            image_name_tensor.append(scalarimage)
                            del (scalarimage)
                    nplist = self.augornot(image_name_tensor, i)
                    nplist_list.append(nplist)
                    del (nplist)
                    end = time.time()

            del (image_name_tensor)
        if pred:
            self.model_list.clear()
            return nplist_list,asclist_list
        else:
            return nplist_list, None

    def zscore_regression_value(self):
        new_arr = []
        for i,col_name in enumerate(self.clinical_data.columns):
            arr = np.array([self.clinical_data.iloc[:,i]])
            arr = np.transpose(arr,(1,0))
            zscore = StandardScaler()
            arr = zscore.fit_transform(arr)
            new_arr.append(arr)

        new_arr = np.concatenate(new_arr,axis=1)

        return pd.DataFrame(new_arr,index = self.clinical_data.index, columns=self.clinical_data.columns)

    def generate_batch_gradcam(self,mask,model_path,batch_img,batch_cl,clinical_num=13):
        if clinical_num==0:
            return None
        batch_len = len(batch_img)
        mask = re_sample(mask,(32,128,128),'arr').numpy()
        clinical_wise = []
        for j in range(clinical_num):
            model = self.model_list[j]
            model.eval()
            batch_wise = []
            img = batch_img
            cl = batch_cl
            cam_img = self.asc.get_grad_cam(model,img)
            cam_img = cam_img*mask
            cam_img = cam_img * cl[j]
            batch_wise.append(cam_img)
            clinical_wise.append(np.concatenate(tuple(batch_wise),axis=0))
        return np.concatenate(tuple(clinical_wise),axis=1)

    def idx_augmentation(self, idx):
        image_idx = self.image_idx_list.iloc[self.idx_list[idx], 0]

        nplist = []
        image_name_tensor = {}
        croporpadtrans = tio.CropOrPad(TARGET_SIZE)

        if (idx % self.aug_time != 0):
            scales = random.uniform(0.8, 1.2)
            degree = random.randint(-10, 10)
            trans = random.randint(-10, 10)
            for image_name in self.image_list:
                image_file = os.path.join(self.image_path, str(image_idx), image_name)
                image = composetrans(image_file, scales, degree, trans).data.numpy()
                image = np.transpose(image, (0, 3, 1, 2))
                nplist.append(image[0].tolist())
        else:
            for image_name in self.image_list:
                image_file = os.path.join(self.image_path, str(image_idx), image_name)
                image = croporpadtrans(tio.ScalarImage(image_file)).data.numpy()
                image = np.transpose(image, (0, 3, 1, 2))
                nplist.append(image[0].tolist())

        self.np_list[idx] = nplist.copy()

    def augornot(self, image_name_tensor, idx):
        nplist = []
        # print(idx,end=' ')
        if (idx % self.aug_time != 0):
            scales = random.uniform(0.8, 1.2)
            degree = random.randint(-10, 10)
            trans = random.randint(-10, 10)
            for i, image_name in enumerate(image_name_tensor):
                image = composetrans(image_name_tensor[i], scales, degree, trans).data.numpy()
                image = np.transpose(image, (0, 3, 1, 2))
                nplist.append(image[0])
                del image
        else:
            for i, image_name in enumerate(image_name_tensor):
                image = croporpadtrans(image_name_tensor[i]).data.numpy()
                image = np.transpose(image, (0, 3, 1, 2))
                nplist.append(image[0])
                del (image)

        return np.array(nplist)

    def all_save_load_augmentation(self):
        print('Aug')
        nplist_list = []

        name_id = 0
        last_idx = -1
        weight = 0

        for idx in self.ori_idx_list:
            print(idx)
            if last_idx == idx:
                weight += 1
            else:
                weight = 0
            last_idx = idx
            image_idx = self.image_idx_list.iloc[idx, 0]
            image_name_tensor = []
            for i in range(self.aug_time * weight, self.aug_time * (weight + 1)):
                print('start', end=' ')
                start = time.time()
                if (image_name_tensor == []):
                    for image_name in self.image_list:
                        image_file = os.path.join(self.image_path, str(image_idx), image_name)
                        scalarimage = tio.ScalarImage(image_file)
                        image_name_tensor.append(scalarimage)
                        del (scalarimage)
                nplist = self.augornot(image_name_tensor, i)
                # nplist_list.append(nplist)
                np.save(self.temp_path + str(name_id) + '.npy', nplist)
                name_id += 1
                del (nplist)
                end = time.time()
                print(end - start)

            del (image_name_tensor)

        return None


