from trainer import RegressionTraining
from interpreter import Interpreting
from models.mate_net import *
from utils.model_load import load_model
from utils.evaluation import reg_evaluation,prc_evaluation
from utils.resample import re_sample
from utils.data_load import HybridDataset

import os
import copy
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import json
import shap

from tqdm import tqdm

torch.backends.cudnn.enabled = False



class Model():
	def __init__(self,jsonfile,devices,task='train',split_num = 0.8,
			  image_list = ['AP.nii', 'AP2.nii', 'CTP0.nii', 'CBV.nii.gz', 'CBF.nii.gz', 'Tmax.nii.gz', 'mttv.nii.gz'],
			  aug_time=10,batch_size=32,num_worker=4):
		self.task = task
		if task == 'train':
			self.trainer = RegressionTraining(
							jsonfile=jsonfile,
							train_test_split_num=split_num,
							image_list=image_list,
							aug_time=aug_time,
							gpu_devices=devices,
							batch_size=(int)(batch_size/len(devices.split(','))),
							num_workers=num_worker)


		self.jsonfile = jsonfile
		self.image_list = image_list
		self.devices = devices
    
	def kfold_train(self,attn_model,pred_model,epoch,value_path,start_idx,training_type,include_list=None,clinical_num=13,kvalue=5):
		if self.task == 'train':
			for i in range(start_idx,start_idx+kvalue):
				if not os.path.exists(os.path.join(value_path,str(i))):
					os.mkdir(os.path.join(value_path,str(i)))

			self.trainer.kfold_training(
				attnmodel=attn_model,
				predmodel=pred_model,
				EPOCH=epoch,
				value_path=value_path,
				start_idx = start_idx,
				clinical_num=clinical_num,
				training_type=training_type,
				include_list=list(range(clinical_num)) if include_list is None else include_list,
				kvalue=kvalue
			)
		else:
			print('Not in training stage.')

	def evaluate(self,attn_model,pred_model,value_path,training_type,start_idx,result_path,kvalue=5):
		image_list = self.image_list
		jsonfile = self.jsonfile

		with open(jsonfile, 'r', encoding='utf8') as fp:
			data = json.load(fp)
		IMAGE_PATH = data['IMAGE_PATH']
		NORM_IMAGE_PATH = data['NORM_IMAGE_PATH']
		CLINICAL_PATH = data['CLINICAL_PATH']
		RESULT_PATH = data['RESULT_PATH']
		IMAGE_IDX_PATH = data['IMAGE_IDX_PATH']
		DATA_LENGTH = data['DATA_LENGTH']
		IMAGE_LIST = image_list
		modelpath = 'best_'+training_type + '_model.pkl'

		idx_list = list(range(DATA_LENGTH))


		for idx_ in range(start_idx,start_idx+kvalue):
			model_idx = str(idx_)
			model = copy.deepcopy(pred_model)
			model = load_model(model,os.path.join(value_path, model_idx, modelpath))
			split_path = os.path.join(value_path, model_idx+'/split_list.txt')
			with open(split_path) as f:
				str1 = f.readline()[1:-2]
				str2 = f.readline()[1:-1]
				list1 = str1.replace(',', ' ').split()
				list2 = str2.replace(',', ' ').split()
				list1 = [int(i) for i in list1]
				list2 = [int(i) for i in list2]
				print(list1, list2)


			test_dataset = HybridDataset(
							idx_list = idx_list,
							image_idx_list=IMAGE_IDX_PATH,
							image_path=IMAGE_PATH,
							norm_image_path=NORM_IMAGE_PATH,
							image_list=IMAGE_LIST,
							clinical_path=CLINICAL_PATH,
							result_path=RESULT_PATH,
							attn_model=attn_model,
							aug_time=1,
							pred=True,
							model_path=os.path.join(value_path,model_idx)
						)

			test_loader = Data.DataLoader(
							dataset=test_dataset,
							batch_size=32,num_workers=4)


			net = torch.nn.DataParallel(model, device_ids=[0])
			net.eval()
			net = net.cuda()
			test_target = []
			test_preds = []
			

			for idx,i in enumerate(tqdm(test_loader)):
				img = re_sample(i['image'],(8,32,32)).cuda()
				asced = re_sample(i['asced'],(8,32,32)).cuda()
				clinical = i['clinical'].cuda()
				target = i['result'].cuda()
				with torch.no_grad():
					test_target.append(target)
					test_preds.append(net(img, asced))

			test_target = torch.cat(test_target, dim=0).cpu()
			test_preds = torch.cat(test_preds, dim=0).cpu()

			file = os.path.join(value_path,model_idx,result_path)
			evas = reg_evaluation(test_preds,test_target)
			prcs = prc_evaluation(test_preds,test_target)
			with open(file, 'w') as f:
				for eva in evas:
					f.write(str(eva)+'\t')
				for prc in prcs:
					f.write(str(prc)+'\t')
				f.write('\n')

	def interprete(self,attn_model,pred_model,value_path,start_idx,training_type,kvalue=1,clinical_num=13,include_list=None,exterjsonfile=None,exter_image_list=None):
		for idx in range(start_idx,start_idx+kvalue):
			inter = Interpreting(
				attn_model=attn_model,
				value_path=value_path,
				idx=idx,
				jsonfile=self.jsonfile,
				devices=self.devices,
				clinical_num=clinical_num,
				image_list=self.image_list,
				include_list=include_list,
				exterjsonfile=exterjsonfile,
				exter_image_list=exter_image_list
			)
			inter.shap_interprete(
				pred_model=pred_model,
				training_type=training_type,
			)

	