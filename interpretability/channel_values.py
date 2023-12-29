import numpy as np
import shap
import os
import torchio as tio
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

from utils.case_load import sample_load,sample_one_img_load


def up_sample(arr,aim_shape):
    ori_shape = arr.shape
    arr = torch.Tensor(arr.reshape(1, 1, *ori_shape))
    arr = F.interpolate(arr, size=aim_shape, mode='nearest')
    arr = arr.numpy().reshape(aim_shape)
    return arr


def area_mean(arr,mask):
    #print(np.sum(mask))
    return np.sum(arr*mask)/np.sum(mask)


def generate_channel_shap_values(arr_path,image_path,df,data_list,extra_image_path=None,extra_df=None,extra_data_list=None,mask_type='mask.nii'):
    scale = 8*32*32
    arr = np.load(arr_path)
    res_arr = []

    all_data_list = data_list + extra_data_list if extra_data_list is not None else data_list
    print(all_data_list)
    print(len(all_data_list),len(arr))
    #mask_list = []
    for i,v0 in enumerate(arr):
        if i<len(data_list) and extra_data_list is not None:
            image_path=image_path
            df = df
        else:
            image_path=extra_image_path
            df = extra_df
        print(all_data_list[i],end=' ')
        mask = sample_one_img_load(df.iloc[all_data_list[i],0], image_path, mask_type)
        aim_shape = mask.shape
        case_arr = []
        for v in v0:
            v = up_sample(v,aim_shape)
            v = area_mean(v,mask)
            case_arr.append(v)
        res_arr.append(np.array(case_arr))
    return np.array(res_arr)*scale*0.01

def image_to_value(image_path,clinical_path,result_path,df,data_list,image_list,mask_type='mask.nii'):
    res_arr = []
    for i in tqdm(range(len(data_list))):
        mask = sample_one_img_load(df.iloc[data_list[i],0], image_path, mask_type)
        sample = sample_load(df.iloc[data_list[i],0],image_path,clinical_path,result_path,image_list,return_type='array')
        images, clinical = sample["image"][0],sample["clinical"][0]
        #print(clinical.shape)
        brain_area = mask==1
        case_arr = []
        for img in images:
            value_arr = img[brain_area]
            #print(value_arr.shape)
            value_arr = (value_arr-np.min(value_arr))/(np.max(value_arr)-np.min(value_arr))
            case_arr.append(np.mean(value_arr))
        case_arr+=list(clinical)
        #print(case_arr)
        res_arr.append(np.array(case_arr))
    return res_arr

def generate_channel_values(idx,k,value_path,result_path,name1,image_path,clinical_path,df,data_list,image_list,extra_result_path=None,
                            name2=None,extra_image_path=None,extra_clinical_path=None,extra_df=None,extra_data_list=None,extra_image_list=None):
    shap_values_list = []
    for i in range(idx,idx+k):


        shap_values = np.array(generate_channel_shap_values(os.path.join(value_path , str(i) , name1), image_path,df,data_list,extra_image_path,
                                                          extra_df,extra_data_list))
        if name2 is not None:
            shap_values = np.concatenate(
                (shap_values,np.array(generate_channel_shap_values(os.path.join(value_path , str(i) , name2), image_path,df,data_list,extra_image_path,
                                                          extra_df,extra_data_list))), axis=1)
        shap_values_list.append(shap_values)
    shap_values = np.concatenate(shap_values_list,axis=0)
    feature_values = np.array(image_to_value(image_path,clinical_path,result_path,df,data_list,image_list))
    if name2 is not None:
        feature_values = np.concatenate(
            (feature_values,np.array(image_to_value(extra_image_path,extra_clinical_path,extra_result_path,extra_df,extra_data_list,extra_image_list))),axis=0)
    feature_values = np.concatenate([feature_values]*k,axis=0)
    data_length = len(data_list) if name2 is None else len(data_list+extra_data_list)
    data_idx_list = random.shuffle(list(range(data_length)))
    shap_values = shap_values[data_idx_list][0]
    feature_values = feature_values[data_idx_list][0]
    return shap_values,feature_values