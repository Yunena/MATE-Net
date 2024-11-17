from models.mate_net import *
from utils.model_save import dataparalell_save
from utils.data_load import MATENetDataset

from utils.early_stop import EarlyStop
from utils.model_load import load_model
import torchmetrics
import os
import torch
import torch.utils.data as Data
import torch.nn as nn
import json
import shap
import numpy as np

from tqdm import tqdm
from torch.optim import lr_scheduler
torch.backends.cudnn.enabled = False



class Model():
	def __init__(self,device,save_path = None,model_type = 'pred',is_evaluate = False):
		assert model_type in ['attn','pred']
		self.aim_shape = (8,32,32)
		self.device = device
		self.save_path = save_path
		self.model_type = model_type
		self.clinical_idx = 0
		self.is_evaluate = is_evaluate
		self.es = EarlyStop()

		pass

	def get_dataloader(self,image_idx_list,image_path,mask_path,minmax_clinical_data,clinical_num,device,
					attn_model_path=None,zscored_clinical_data=None,pred_result_data=None,
					batch_size = 12,num_workers=4,shuffle=False):
		data_set = MATENetDataset(
			model_type=self.model_type,
			image_idx_list=image_idx_list,
			image_path = image_path,
			mask_path = mask_path,
			minmax_clinical_data = minmax_clinical_data,
			clinical_num=clinical_num,
			device=device,
			attn_model_path=attn_model_path,
			zscored_clinical_data = zscored_clinical_data,
			pred_result_data=pred_result_data
		)
		data_loader = Data.DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
		return data_loader
		

	def train(self,epoch,model,data_loader,eval_data_loader = None):
		device = self.device
		model = torch.nn.DataParallel(model, device_ids=[0])
		self.es.initialize()
		if self.model_type=='attn':
			loss_func = nn.MSELoss()
			opti = torch.optim.Adam(model.parameters(), lr=1e-6)
		else:
			loss_func = nn.SmoothL1Loss()
			opti = torch.optim.Adam(model.parameters(), lr=1e-5)

		for e in range(epoch):
			train_loss, predictions, results = self.train_one_epoch(model,data_loader,opti,loss_func,device)
			if self.model_type=='pred':
				print(train_loss,self.evaluate(predictions,results))
			if eval_data_loader is not None:
				test_loss, predictions, results = self.validate_in_train(model,eval_data_loader,loss_func,device)
				if self.model_type=='attn' and self.es.attn_check_stop(test_loss):
					dataparalell_save(model,os.path.join(self.save_path,'attn_module'+'_'+str(self.clinical_idx)+'.pkl'))
				elif self.model_type == 'pred' and self.es.check_stop(test_loss):	
					print(test_loss,self.evaluate(predictions,results))
					dataparalell_save(model,os.path.join(self.save_path,'pred_module.pkl'))
					np.save(os.path.join(self.save_path,'validate_predictions.npy'),predictions.detach().cpu().numpy())
			else:
				if self.model_type=='attn':
					dataparalell_save(model,os.path.join(self.save_path,'attn_module'+'_'+str(self.clinical_idx)+'.pkl'))
				elif self.model_type == 'pred':	
					dataparalell_save(model,os.path.join(self.save_path,'pred_module.pkl'))

	def validate(self,model,data_loader,device):
		model = load_model(model,os.path.join(self.save_path,'pred_module.pkl'))
		model = torch.nn.DataParallel(model, device_ids=[0])
		model.to(device)
		model.eval()
		predictions = []
		results = []
		for step,sample in enumerate(data_loader):
			i_x = sample["image"]
			if self.model_type == 'pred':
				r_x = sample["result"]
				results.append(r_x)
				i_x, r_x = i_x.to(device), r_x.to(device)
				c_x = sample["asced"]
				c_x = c_x.to(device)
				o_x = model(i_x,c_x)
			else:
				r_x = sample["result"][:,self.clinical_idx].unsqueeze(1)
				results.append(r_x)
				i_x, r_x = i_x.to(device), r_x.to(device)
				o_x = model(i_x)
			predictions.append(o_x.cpu())
		predictions = torch.cat(predictions)
		results = torch.cat(results)
		if self.is_evaluate:
			print(self.evaluate(predictions,results))
		return predictions, results

	def validate_in_train(self,model,data_loader,loss_func,device):
		model.eval()
		total_loss = 0
		predictions = []
		results = []
		for step,sample in enumerate(data_loader):
			i_x = sample["image"]
			if self.model_type == 'pred':
				r_x = sample["result"]
				results.append(r_x)
				i_x, r_x = i_x.to(device), r_x.to(device)
				c_x = sample["asced"]
				c_x = c_x.to(device)
				o_x = model(i_x,c_x)
			else:
				r_x = sample["result"][:,self.clinical_idx].unsqueeze(1)
				results.append(r_x)
				i_x, r_x = i_x.to(device), r_x.to(device)
				o_x = model(i_x)
			loss = loss_func(o_x, r_x)
			predictions.append(o_x.cpu())
			total_loss += float(loss.data.cpu().numpy())
		predictions = torch.cat(predictions)
		results = torch.cat(results)
		if self.is_evaluate:
			print(self.evaluate(predictions,results))
		return total_loss/(step+1), predictions, results

	def train_one_epoch(self,model,data_loader,optimizer,loss_func,device):
		model.train()
		total_loss = 0
		predictions = []
		results = []
		for step,sample in enumerate(tqdm(data_loader)):
			i_x = sample["image"]
			if self.model_type == 'pred':
				r_x = sample["result"]
				results.append(r_x)
				i_x, r_x = i_x.to(device), r_x.to(device)
				c_x = sample["asced"]
				c_x = c_x.to(device)
				o_x = model(i_x,c_x)
			else:
				r_x = sample["result"][:,self.clinical_idx].unsqueeze(1)
				results.append(r_x)
				i_x, r_x = i_x.to(device), r_x.to(device)
				o_x = model(i_x)
			loss = loss_func(o_x, r_x)
			loss.backward()
			predictions.append(o_x.cpu())
			optimizer.step()
			total_loss += float(loss.data.cpu().numpy())
		predictions = torch.cat(predictions)
		results = torch.cat(results)
		return total_loss/(step+1), predictions, results

	def evaluate(self,predictions,results):
		predictions = predictions.cpu()
		results = results.cpu()
		predictions = torch.round(predictions).int()
		float_predictions = torch.round(predictions)/6
		results = results.int()
		predictions[predictions<3]=0
		predictions[predictions>2]=1
		results[results<3] = 0
		results[results>2] = 1
		biacc = torchmetrics.Accuracy('binary')
		auc = torchmetrics.AUROC(task = 'binary')
		pre = torchmetrics.Precision(task = 'binary')
		cm = torchmetrics.ConfusionMatrix(task = 'binary',num_classes=2)
		biacc_r = biacc(predictions,results)
		auc_r = auc(float_predictions,results)
		ppv_r = pre(predictions,results)
		cm_r = cm(predictions,results)
		npv_r = cm_r[0][0]/torch.sum(cm_r,axis=0)[0] if cm_r[0][0]!=0 else cm_r[0][0]


		return biacc_r, auc_r, ppv_r, npv_r

	def interprete(self,model,base_loader,data_loader):
		device = self.device
		model = torch.nn.DataParallel(model, device_ids=[0])
		model.eval()
		model = model.to(device)
		x = None
		for base in base_loader:
			x = [base["image"].to(device),base["asced"].to(device)]
		explainer = shap.DeepExplainer(model, x)
		values = []
		for data in data_loader:
			x = [data["image"].to(device),data["asced"].to(device)]
			value = explainer.shap_values(x)
			values.append(np.concatenate(value,axis=1))
		return np.concatenate(values,axis=0)
		


	