import numpy as np
import os

def generate_shaplist(shap_arr,ch_idx_list):
    shap_list = []
    for ch_idx in ch_idx_list:
        arr = shap_arr[:,ch_idx:ch_idx+1]
        shap_list.append(arr)
    return shap_list

def generate_cshap(shap_list):
    cshap_list = []
    for arr in shap_list:
        arr = (arr-np.mean(arr))/np.std(arr)
        cshap_list.append(arr)
    return np.sum(np.concatenate(cshap_list,axis=1),axis=1)