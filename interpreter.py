import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import random


#import torch
import torch.nn as nn
import torch.utils.data as Data
import json


from utils.model_load import load_model
from utils.data_load import HybridDataset
from utils.resample import re_sample
from utils.data_load import HybridDataset

from interpretability.channel_values import generate_channel_values
from interpretability.cshap_generate import *


torch.autograd.set_detect_anomaly(True)



class Interpreting:
    def __init__(self,attn_model,value_path,idx,jsonfile,devices,clinical_num=13,
                 image_list=['AP.nii', 'AP2.nii', 'CTP0.nii', 'CBV.nii.gz', 'CBF.nii.gz', 'Tmax.nii.gz', 'mttv.nii.gz'],
                 inte_type='interprete',include_list=None,exterjsonfile=None,exter_image_list=None):
        assert inte_type in ['interprete','channel','cshap']
        model_path = os.path.join(value_path,str(idx))
        self.value_path = model_path
        self.devices = devices
        split_list_path = os.path.join(value_path, str(idx), 'split_list.txt')
        self.baseline = None
        self.aimloader = None
        self.jsonfile = jsonfile
        self.exterjsonfile = exterjsonfile
        self.image_list = image_list
        self.exter_image_list = exter_image_list
        self.idx = idx
        with open(split_list_path) as f:
            str1 = f.readline()[1:-2]
            str2 = f.readline()[1:-1]
            list1 = str1.replace(',', ' ').split()
            list2 = str2.replace(',', ' ').split()
            list1 = [int(i) for i in list1]
            list2 = [int(i) for i in list2]
            print(list1, list2)
        train_list = list1
        
        
        if inte_type=='interprete':
            self.generate_interprete_loader(attn_model,model_path,jsonfile,train_list,clinical_num,include_list,exterjsonfile)
    
    def generate_interprete_loader(self,attn_model,model_path,jsonfile,train_list,clinical_num=13,include_list=[],exterjsonfile=None):
        with open(jsonfile, 'r', encoding='utf8') as fp:
            data = json.load(fp)        
        clinical_path = data["CLINICAL_PATH"]
        result_path = data["RESULT_PATH"]
        image_path = data["IMAGE_PATH"]
        norm_image_path = data["NORM_IMAGE_PATH"]
        image_idx_list = data["IMAGE_IDX_PATH"]
        image_list = image_list

        if exterjsonfile is not None:
            with open(exterjsonfile, 'r', encoding='utf8') as fp:
                data = json.load(fp)        
            exter_clinical_path = data["CLINICAL_PATH"]
            exter_result_path = data["RESULT_PATH"]
            exter_image_path = data["IMAGE_PATH"]
            exter_norm_image_path = data["NORM_IMAGE_PATH"]
            exter_image_idx_list = data["IMAGE_IDX_PATH"]
            exter_image_list = exter_image_list
        train_dataset = HybridDataset(
            idx_list=train_list,
            image_idx_list=image_idx_list,
            image_path=image_path,
            norm_image_path=norm_image_path,
            image_list=image_list,
            clinical_path=clinical_path,
            result_path=result_path,
            attn_model=attn_model,
            aug_time=1,
            pred=True,
            model_path=model_path,
            clinical_num=clinical_num,
            include_list=list(range(clinical_num)) if include_list is None else include_list
        )
        baseloaders = Data.DataLoader(dataset=train_dataset,batch_size=len(train_list))
        for baseloader in baseloaders:
            self.baseline = baseloader
        
        test_dataset = HybridDataset(
            idx_list=list(range(data["DATA_LENGTH"])),
            image_idx_list=image_idx_list,
            image_path=image_path,
            norm_image_path=norm_image_path,
            image_list=image_list,
            clinical_path=clinical_path,
            result_path=result_path,
            attn_model=attn_model,
            aug_time=1,
            pred=True,
            model_path=model_path,
            clinical_num=clinical_num,
            include_list=list(range(clinical_num)) if include_list is None else include_list
        )
        aimloader = Data.DataLoader(dataset=test_dataset,batch_size=1)
        
        if exterjsonfile is not None:
            exter_dataset = HybridDataset(
                idx_list=list(range(data["DATA_LENGTH"])),
                image_idx_list=exter_image_idx_list,
                image_path=exter_image_path,
                norm_image_path=exter_norm_image_path,
                image_list=exter_image_list,
                clinical_path=exter_clinical_path,
                result_path=exter_result_path,
                attn_model=attn_model,
                aug_time=1,
                pred=True,
                model_path=model_path,
                clinical_num=clinical_num,
                include_list=list(range(clinical_num)) if include_list is None else include_list
            )
            exterloader = Data.DataLoader(dataset=exter_dataset,batch_size=1)
            self.aimloader = (aimloader,exterloader)
        else:
            self.aimloader = (aimloader)

    def shap_interprete(self,pred_model,training_type,is_save=True):
        assert training_type in ["attn","multimodal","image","clinical","hic","fc","abl1","abl2","abl3","abl4","abl5","abl6"]
        baseline = self.baseline
        loaders = self.aimloader
        model_path = os.path.join(self.value_path,'best_'+training_type + '_model.pkl')
        devices = self.devices
        
        model = pred_model
        model.eval()
        model = load_model(model,model_path)
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        model = nn.DataParallel(model, device_ids=[0])

        model.cuda()
        model.eval()
        print('model')

        shap.initjs()
        aim_shape = (8,32,32)
        image = re_sample(baseline["image"],aim_shape)
        asced = re_sample(baseline["asced"],aim_shape)

        if training_type=="multimodal":
            x = [image.cuda(),asced.cuda()]
        elif training_type=="image":
            x = [image.cuda()]
        elif training_type=="clinical":
            x = [asced.cuda()]
        explainer = shap.DeepExplainer(model, x)
        print('start')
        finalvalue = []

        if training_type=="multimodal":
            for aims in loaders:
                for sample in aims:
                    image = re_sample(sample["image"],aim_shape)
                    asced = re_sample(sample["asced"],aim_shape)
                    x = [image.cuda(),asced.cuda()]
                    value = explainer.shap_values(x)
                    finalvalue.append(value)
                    print(len(finalvalue))

        elif training_type=="image":
            for aims in loaders:
                for sample in aims:
                    image = re_sample(sample["image"],aim_shape)
                    x = [image.cuda()]
                    value = explainer.shap_values(x)
                    finalvalue.append(value)
                    print(len(finalvalue))
                    
        elif training_type=="clinical":
            for aims in loaders:
                for sample in aims:
                    asced = re_sample(sample["asced"],aim_shape)
                    x = [asced.cuda()]
                    value = explainer.shap_values(x)
                    finalvalue.append(value)
                    print(len(finalvalue))


        if is_save:
            finalvalue = list(zip(*finalvalue))
            for j in range(len(finalvalue)):
                value = np.concatenate(tuple(finalvalue[j]), axis=0)
                np.save(self.value_path+'/'+training_type+'_values'+str(j)+'.npy', value)
                print(np.shape(value))

    def get_channel_values(self,name1,value_path,training_type,name2=None,k=5,is_save = True):
        with open(self.jsonfile, 'r', encoding='utf8') as fp:
            data = json.load(fp)        
        clinical_path = data["CLINICAL_PATH"]
        result_path = data["RESULT_PATH"]
        image_path = data["IMAGE_PATH"]
        image_idx_path = data["IMAGE_IDX_PATH"]
        image_list = self.image_list
        df = pd.read_excel(image_idx_path, header=0, index_col=None)
        data_list = list(range(data["DATA_LENGTH"]))

        if self.exterjsonfile is not None:
            with open(self.exterjsonfile, 'r', encoding='utf8') as fp:
                data = json.load(fp)        
            exter_clinical_path = data["CLINICAL_PATH"]
            exter_result_path = data["RESULT_PATH"]
            exter_image_path = data["IMAGE_PATH"]
            exter_image_idx_path = data["IMAGE_IDX_PATH"]
            exter_image_list = self.exter_image_list
            exter_df = pd.read_excel(exter_image_idx_path, header=0, index_col=None)
            exter_data_list = list(range(data["DATA_LENGTH"]))
        
            shap_values,feature_values = generate_channel_values(
                self.idx,k,value_path,result_path,name1,image_path,clinical_path,df,data_list,image_list,exter_result_path,
                                name2,exter_image_path,exter_clinical_path,exter_df,exter_data_list,exter_image_list
            )
        else:
            shap_values,feature_values = generate_channel_values(
                self.idx,k,value_path,result_path,name1,image_path,clinical_path,df,data_list,image_list,name2=name2
            )

        if is_save:
            np.save(os.path.join(value_path,training_type+'_shap_values.npy'),shap_values)
            np.save(os.path.join(value_path,training_type+'_feature_values.npy'),feature_values)

    def get_cshap(self,name1,save_path,id_list=[5,7,10],name2=None):
        shap_list = []
        for i in range(10,15):
            shap_list_ = []
            filepath = os.path.join(self.value_path,str(i),name1)
            shap_list_.append(np.load(filepath))
            if name2 is not None:
                filepath = os.path.join(self.value_path,str(i),name2)
                shap_list_.append(np.load(filepath))
                shap_list.append(np.concatenate(tuple(shap_list_),axis=1))
            else:
                shap_list.append(shap_list_)
        shap_arr = np.concatenate(tuple(shap_list),axis=0)
        shap_list = generate_shaplist(shap_arr,id_list)
        cshap_arr = generate_cshap(shap_list)
        np.save(save_path,cshap_arr)
