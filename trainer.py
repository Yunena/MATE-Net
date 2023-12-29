from utils.data_load import HybridDataset
from utils.model_load import load_model
from utils.model_save import dataparalell_save
from utils.evaluation import reg_evaluation
from utils.early_stop import EarlyStop
from utils.resample import re_sample
from models.dimension_ascend import Ascender
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as Data



import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
import copy

import os
import random
import json



class RegressionTraining():
    frozen_flag = False

    def __init__(self, jsonfile, train_test_split_num, image_list, aug_time, gpu_devices, pin_memory=False,
                 batch_size=12, num_workers=4, imbalanced_weight=1):
        data = None
        self.flag = False
        with open(jsonfile, 'r', encoding='utf8') as fp:
            data = json.load(fp)
        self.JSONFILE = jsonfile
        RESULT_PATH = data['RESULT_PATH']
        self.result = pd.read_excel(RESULT_PATH, header=0, index_col=0)
        DATA_LENGTH = data['DATA_LENGTH']
        self.CLINICAL_NUM = data['CLINICAL_NUM']
        ORI_IDX_LIST = [x for x in range(DATA_LENGTH)]
        IDX0, IDX1 = self.stratified_random(ORI_IDX_LIST)
        self.IDX0 = np.array(IDX0)
        self.IDX1 = np.array(IDX1)
        IDX0_SPLIT = int(len(IDX0) * train_test_split_num)
        IDX1_SPLIT = int(len(IDX1) * train_test_split_num)

        self.image_list = image_list
        self.aug_time = aug_time
        DEVICES = gpu_devices
        self.devices_num = len(DEVICES.split(','))
        BATCH_SIZE = batch_size * self.devices_num
        NUM_WORKERS = num_workers * self.devices_num
        PIN_MEMORY = pin_memory
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.pin_memory = PIN_MEMORY
        self.es = EarlyStop()

        self.TRAIN_LOADER = None
        self.TEST_LOADER = None
        self.train_list = IDX0[:IDX0_SPLIT] + IDX1[:IDX1_SPLIT] * imbalanced_weight
        self.test_list = IDX0[IDX0_SPLIT:] + IDX1[IDX1_SPLIT:] * imbalanced_weight
        self.asc = Ascender(layer='model.layer1')


        self.LR = data['LR']

        self.TRIAN_RESULTS_LIST=[]
        self.TEST_RESULTS_LIST=[]
        self.RESULTS_NAME_LIST =['loss','acc','r2','binary','auc','fpr','tpr','pre','rec','spe','npv','f1']
        for name in self.RESULTS_NAME_LIST:
            self.TRIAN_RESULTS_LIST.append([])
            self.TEST_RESULTS_LIST.append([])

    def get_generated_list(self,split_list_path,pred=False,model_path=None):
        if os.path.exists(split_list_path):
            with open(split_list_path) as f:
                str1 = f.readline()[1:-2]
                str2 = f.readline()[1:-1]
                list1 = str1.replace(',', ' ').split()
                list2 = str2.replace(',', ' ').split()
                list1 = [int(i) for i in list1]
                list2 = [int(i) for i in list2]
                print(list1, list2)
            self.train_list = list1
            self.test_list = list2
        with open(self.JSONFILE, 'r', encoding='utf8') as fp:
            data = json.load(fp)
        IMAGE_PATH = data['IMAGE_PATH']
        NORM_IMAGE_PATH = data['NORM_IMAGE_PATH']
        CLINICAL_PATH = data['CLINICAL_PATH']
        RESULT_PATH = data['RESULT_PATH']
        IMAGE_IDX_PATH = data['IMAGE_IDX_PATH']
        IMAGE_LIST = self.image_list
        AUG_TIME = self.aug_time
        BATCH_SIZE = self.batch_size
        NUM_WORKERS = self.num_workers
        PIN_MEMORY = self.pin_memory
        self.TRAIN_LOADER,self.TEST_LOADER = self.generate_dataloader(IMAGE_IDX_PATH,IMAGE_PATH,NORM_IMAGE_PATH, IMAGE_LIST,CLINICAL_PATH,RESULT_PATH, 
                                                                      AUG_TIME,BATCH_SIZE,NUM_WORKERS,PIN_MEMORY,pred,model_path)

    def generate_dataloader(self, IMAGE_IDX_PATH, IMAGE_PATH, NORM_IMAGE_PATH, IMAGE_LIST, CLINICAL_PATH, RESULT_PATH, 
                            AUG_TIME, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, attn_model, pred=False,model_path=None,
                            clinical_num=13,include_list = None):
        train_dataset = HybridDataset(
            idx_list=self.train_list,
            image_idx_list=IMAGE_IDX_PATH,
            image_path=IMAGE_PATH,
            norm_image_path=NORM_IMAGE_PATH,
            image_list=IMAGE_LIST,
            clinical_path=CLINICAL_PATH,
            result_path=RESULT_PATH,
            attn_model=attn_model,
            aug_time=AUG_TIME,
            pred=pred,
            model_path=model_path,
            clinical_num=clinical_num,
            include_list=include_list
        )

        test_dataset = HybridDataset(
            idx_list=self.test_list,
            image_idx_list=IMAGE_IDX_PATH,
            image_path=IMAGE_PATH,
            norm_image_path=NORM_IMAGE_PATH,
            image_list=IMAGE_LIST,
            clinical_path=CLINICAL_PATH,
            result_path=RESULT_PATH,
            attn_model=attn_model,
            aug_time=1,
            pred=pred,
            model_path=model_path,
            clinical_num=clinical_num,
            include_list=include_list
        )

        TRAIN_LOADER = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
            pin_memory=PIN_MEMORY
        )

        TEST_LOADER = Data.DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            pin_memory=PIN_MEMORY
        )
        return TRAIN_LOADER,TEST_LOADER

    def attn_training(self, model,  EPOCH, train_value_path, test_value_path, train_result_path, test_result_path,
                        for_idx, image_path, model_path,train_loader=None,test_loader=None,clinical_idx = 0):
        with open(train_value_path,'w') as f:
            pass
        with open(test_value_path,'w') as f:
            pass
        train_loader = self.TRAIN_LOADER if train_loader==None else train_loader
        test_loader = self.TEST_LOADER if test_loader==None else test_loader
        net = model
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(self.devices_num)])

        #opti = torch.optim.Adam(net.parameters(), lr=self.LR*0.0001)
        #opti = torch.optim.Adam(net.parameters(), lr=self.LR * 0.001)
        #opti = torch.optim.Adam(net.parameters(), lr=self.LR)
        opti = torch.optim.Adam(net.parameters(), lr=self.LR * 0.01)
        sche = lr_scheduler.ReduceLROnPlateau(opti, mode='min', factor=0.1, patience=3)


        loss_func = nn.MSELoss()
        test_loss_func = nn.MSELoss()

        net = net.cuda()
        loss_func = loss_func.cuda()
        test_loss_func = test_loss_func.cuda()

        for epoch in range(EPOCH):

            print('EPOCH:', epoch)

            net.train()
            train_loss = 0

            for step, sample in enumerate(tqdm(train_loader)):
                #print('start')
                i_x = sample["image"].cuda()
                c_x = sample["normalized"][:,clinical_idx:clinical_idx+1].cuda()
                output= net(i_x)

                loss = loss_func(output, c_x)
                train_loss += loss.data.cpu().numpy()
                opti.zero_grad()
                loss.backward()
                opti.step()

                

            train_loss /= (step+1)


            with open(train_value_path,'a') as f:
                f.write(str(train_loss))
                f.write('\n')

            test_loss = 0


            net.eval()
            for step, sample in enumerate(test_loader):
                i_x = sample["image"].cuda()
                c_x = sample["normalized"][:,clinical_idx:clinical_idx+1].cuda()
                with torch.no_grad():
                    test_output = net(i_x)
                    test_loss += test_loss_func(test_output,c_x).data.cpu().numpy()

                #del(ca)
                #del(sa)
            test_loss /= (step+1)
            sche.step(train_loss)
            with open(test_value_path,'a') as f:
                f.write(str(test_loss))
                f.write('\n')
            if self.es.attn_check_stop(test_loss):
                dataparalell_save(net,model_path)
            print('Epoch:', epoch, '\ttrain loss:', np.mean(train_loss), '\ttest loss:',np.mean(test_loss))
            torch.cuda.empty_cache()
            if self.es.early_stop:
                break
        self.es.initialize()

    def training(self, attn_model,model, EPOCH, train_value_path, test_value_path, train_result_path, test_result_path,
                        for_idx, image_path, model_path, dir_path, training_type, train_best_path = None,test_best_path = None,
                        model_best_path = None,train_loader=None,test_loader=None):
        assert training_type in ["multimodal","image","clinical","abl1","abl2","abl3","abl4","abl5","abl6"]
        with open(train_value_path,'w') as f:
            pass
        with open(test_value_path,'w') as f:
            pass
        train_loader = self.TRAIN_LOADER if train_loader==None else train_loader
        test_loader = self.TEST_LOADER if test_loader==None else test_loader
        net = model
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(self.devices_num)])
        opti = torch.optim.Adam(net.parameters(), lr=self.LR*0.1) #the best lr without z-score
        sche = lr_scheduler.ReduceLROnPlateau(opti, mode='min', factor=0.8, patience=3)

        loss_func = nn.SmoothL1Loss()
        test_loss_func = nn.SmoothL1Loss()


        net = net.cuda()
        loss_func = loss_func.cuda()
        test_loss_func = test_loss_func.cuda()
        aim_shape = (8,32,32)

        for epoch in range(EPOCH):

            print('EPOCH:', epoch)

            net.train()
            train_pred = []
            train_target = []
            train_loss = 0

            for step, sample in enumerate(tqdm(train_loader)):
                i_x = sample["image"]
                i_x = re_sample(i_x,aim_shape)
                i_x = i_x.cuda()
                #c_x = sample["clinical"]
                b_y = sample["result"].cuda()
                if training_type in ["multimodal","clinical","abl1","abl2","abl3","abl4","abl5","abl6"]:
                    s_a = sample["asced"]
                    s_a = re_sample(s_a,aim_shape)
                    s_a = s_a.cuda()
                else:
                    s_a = None

                if training_type=="clinical":
                    i_x = s_a
                    s_a = None

                pm_out = net(i_x,s_a)
                loss2 = loss_func(pm_out,b_y)
                loss = loss2

                train_loss += (float)(loss.data.cpu().numpy())
                opti.zero_grad()
                loss.backward()
                opti.step()

                train_pred.append(pm_out.cpu().clone())
                train_target.append(b_y.cpu().clone())
            train_loss /= (step+1)
            train_pred = torch.cat(train_pred, dim=0)
            train_target = torch.cat(train_target, dim=0)

            train_results = reg_evaluation(
                                           torch_list=train_pred,
                                           target=train_target)

            with open(train_value_path,'a') as f:
                for v in train_results:
                    f.write(str(v)+'\t')
                f.write(str(train_loss))
                f.write('\n')



            test_loss = 0
            test_preds = []
            test_target = []

            net.eval()
            for step, sample in enumerate(test_loader):
                i_x = sample["image"]
                i_x = re_sample(i_x,aim_shape)
                i_x = i_x.cuda()
                b_y = sample["result"].cuda()
                test_target.append(b_y.clone())
                if training_type in ["multimodal","clinical","abl1","abl2","abl3","abl4","abl5","abl6"]:
                    s_a = sample["asced"]
                    s_a = re_sample(s_a,aim_shape)
                    s_a = s_a.cuda()
                else:
                    s_a = None

                if training_type=="clinical":
                    i_x = s_a
                    s_a = None
                with torch.no_grad():
                    test_pmo = net(i_x,s_a)
                    test_loss2 = test_loss_func(test_pmo,b_y)
                    test_preds.append(test_pmo.clone())
                    test_loss += (float)((test_loss2).data.cpu().numpy())


            test_loss /= (step+1)
            sche.step(test_loss)
            test_target = torch.cat(test_target, dim=0).cpu()
            test_preds = torch.cat(test_preds, dim=0).cpu()

            test_results= reg_evaluation(
                torch_list=test_preds,
                target=test_target
            )
            with open(test_value_path,'a') as f:
                for v in test_results:
                    f.write(str(v)+'\t')
                f.write(str(test_loss))
                f.write('\n')
            dataparalell_save(net,model_path)
            print('Epoch:', epoch, '\ttrain loss:', train_loss, '\ttest loss:',test_loss)

            if self.es.check_stop(train_results, test_results):
                with open(train_best_path, 'w') as f:
                    for v in train_results:
                        f.write(str(v) + '\t')
                    f.write(str(train_loss))
                    f.write('\n')
                with open(test_best_path,'w') as f:
                    for v in test_results:
                        f.write(str(v)+'\t')
                    f.write(str(test_loss))
                    f.write('\n')
                dataparalell_save(net,model_best_path)



            #print(train_results)
            #print(test_results)

            torch.cuda.empty_cache()
        self.es.initialize()

    def hic_training(self, model, EPOCH, train_value_path, test_value_path,
                        model_path, training_type, train_best_path = None,test_best_path = None,
                        model_best_path = None,train_loader=None,test_loader=None):
        assert training_type in ["hic","fc"]
        with open(train_value_path,'w') as f:
            pass
        with open(test_value_path,'w') as f:
            pass
        train_loader = self.TRAIN_LOADER if train_loader==None else train_loader
        test_loader = self.TEST_LOADER if test_loader==None else test_loader
        net = model
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(self.devices_num)])
        if training_type=='hic':
            opti = torch.optim.Adam(net.parameters(), lr=self.LR * 0.0001)
            sche = lr_scheduler.ReduceLROnPlateau(opti, mode='min', factor=0.8, patience=3)
        elif training_type =='fc':
            opti = torch.optim.Adam(net.parameters(), lr=self.LR * 0.5)
            sche = lr_scheduler.ReduceLROnPlateau(opti, mode='min', factor=0.1, patience=10)
            #EPOCH = 100
        #sche = lr_scheduler.StepLR(opti, step_size=10, gamma=0.1)

        loss_func = nn.SmoothL1Loss()
        test_loss_func = nn.SmoothL1Loss()


        net = net.cuda()
        loss_func = loss_func.cuda()
        test_loss_func = test_loss_func.cuda()
        aim_shape = (8,32,32)

        for epoch in range(EPOCH):

            print('EPOCH:', epoch)

            net.train()
            train_pred = []
            train_target = []
            train_loss = 0

            for step, sample in enumerate(tqdm(train_loader)):
                i_x = sample["image"]
                i_x = re_sample(i_x,aim_shape)
                i_x = i_x.cuda()
                c_x = sample["clinical"].cuda()
                b_y = sample["result"].cuda()
                if training_type=="hic":
                    pm_out = net(i_x,c_x)
                elif training_type=="fc":
                    pm_out = net(c_x)
                loss2 = loss_func(pm_out,b_y)
                loss = loss2

                train_loss += (float)(loss.data.cpu().numpy())
                opti.zero_grad()
                loss.backward()
                opti.step()

                train_pred.append(pm_out.cpu().clone())
                train_target.append(b_y.cpu().clone())
            #print(ca,sa)
            train_loss /= (step+1)
            train_pred = torch.cat(train_pred, dim=0)
            train_target = torch.cat(train_target, dim=0)

            train_results = reg_evaluation(
                                           torch_list=train_pred,
                                           target=train_target)
            #print(train_results)

            with open(train_value_path,'a') as f:
                for v in train_results:
                    f.write(str(v)+'\t')
                f.write(str(train_loss))
                f.write('\n')



            test_loss = 0
            test_preds = []
            test_target = []

            net.eval()
            for step, sample in enumerate(test_loader):
                i_x = sample["image"]
                i_x = re_sample(i_x,aim_shape)
                i_x = i_x.cuda()
                c_x = sample["clinical"].cuda()
                b_y = sample["result"].cuda()
                test_target.append(b_y.clone())
                with torch.no_grad():
                    if training_type=="hic":
                        test_pmo = net(i_x,c_x)
                    elif training_type=="fc":
                        test_pmo = net(c_x)
                    test_loss2 = test_loss_func(test_pmo,b_y)
                    test_preds.append(test_pmo.clone())
                    test_loss += (float)((test_loss2).data.cpu().numpy())


            test_loss /= (step+1)
            sche.step(test_loss)
            test_target = torch.cat(test_target, dim=0).cpu()
            test_preds = torch.cat(test_preds, dim=0).cpu()

            test_results= reg_evaluation(
                torch_list=test_preds,
                target=test_target
            )
            with open(test_value_path,'a') as f:
                for v in test_results:
                    f.write(str(v)+'\t')
                f.write(str(test_loss))
                f.write('\n')
            dataparalell_save(net,model_path)
            print('Epoch:', epoch, '\ttrain loss:', train_loss, '\ttest loss:',test_loss)

            if self.es.check_stop(train_results, test_results):
                with open(train_best_path, 'w') as f:
                    for v in train_results:
                        f.write(str(v) + '\t')
                    f.write(str(train_loss))
                    f.write('\n')
                with open(test_best_path,'w') as f:
                    for v in test_results:
                        f.write(str(v)+'\t')
                    f.write(str(test_loss))
                    f.write('\n')
                dataparalell_save(net,model_best_path)



            #print(train_results)
            #print(test_results)

            torch.cuda.empty_cache()
        self.es.initialize()

    def generate_batch_gradcam(self,model_,model_path,batch_img,batch_cl,clinical_num=13):
        batch_len = len(batch_img)
        clinical_wise = []
        for j in range(clinical_num):
            model_ = load_model(model_,os.path.join(model_path,'model'+str(j)+'.pkl'))
            model = model_.cuda()
            model.eval()
            batch_wise = []
            for i in range(batch_len):
                img = batch_img[i:i+1]
                cl = batch_cl[i]
                cam_img = self.asc.get_grad_cam(model,img)
                cam_img = cam_img*cl[j].data.numpy()
                batch_wise.append(cam_img)
            clinical_wise.append(np.concatenate(tuple(batch_wise),axis=0))
        return torch.Tensor(np.concatenate(tuple(clinical_wise),axis=1))

    def stratified_random(self, idx_list, threshold=3):
        result_list = list(self.result.iloc[:, 0])
        # print(result_list)
        bires_list = np.array([0 if i < threshold else 1 for i in result_list])
        # print(bires_list)
        idx_list = np.array(idx_list)
        list0 = list(idx_list[np.where(bires_list == 0)])
        # print(list0)
        list1 = list(idx_list[np.where(bires_list == 1)])
        random.shuffle(list0)
        random.shuffle(list1)
        return list0, list1

    def kfold_training(self, attnmodel,predmodel, EPOCH, value_path, start_idx, clinical_num, 
                       training_type='attn',kvalue=5, include_list = None):
        assert training_type in ["attn","multimodal","image","clinical","hic","fc","abl1","abl2","abl3","abl4","abl5","abl6"]
        IDX0 = self.IDX0
        IDX1 = self.IDX1
        IDX0 = np.array_split(IDX0,10)
        IDX1 = np.array_split(IDX1,10)


        with open(self.JSONFILE, 'r', encoding='utf8') as fp:
            data = json.load(fp)
        IMAGE_PATH = data['IMAGE_PATH']
        NORM_IMAGE_PATH = data['NORM_IMAGE_PATH']
        CLINICAL_PATH = data['CLINICAL_PATH']
        RESULT_PATH = data['RESULT_PATH']
        IMAGE_IDX_PATH = data['IMAGE_IDX_PATH']
        DATA_LENGTH = data['DATA_LENGTH']
        IMAGE_LIST = self.image_list
        AUG_TIME = self.aug_time
        BATCH_SIZE = self.batch_size
        NUM_WORKERS = self.num_workers
        PIN_MEMORY = self.pin_memory


        if not os.path.exists(os.path.join(value_path, str(start_idx), 'split_list.txt')):
            for i in range(kvalue):
                train_idx = np.concatenate(IDX0[0:i * 2] + IDX0[2 * i + 2:10] + IDX1[0:8 - 2 * i] + IDX1[10 - 2 * i:10])
                test_idx = np.concatenate(IDX0[2 * i:2 * i + 2] + IDX1[8 - 2 * i:10 - 2 * i])
                train_idx = list(train_idx)
                test_idx = list(test_idx)
                #print(train_idx, test_idx)
                idx = start_idx + i

                if not os.path.exists(os.path.join(value_path, str(idx))):
                    os.mkdir(os.path.join(value_path, str(idx)))

                with open(os.path.join(value_path, str(idx), 'split_list.txt'), 'w') as f:
                    f.write(str(train_idx) + '\n')
                    f.write(str(test_idx))

        for i in range(kvalue):
            model = copy.deepcopy(predmodel)
            idx = start_idx + i
            split_list_path = os.path.join(value_path, str(idx), 'split_list.txt')
            with open(split_list_path) as f:
                str1 = f.readline()[1:-2]
                str2 = f.readline()[1:-1]
                list1 = str1.replace(',', ' ').split()
                list2 = str2.replace(',', ' ').split()
                train_idx = [int(i) for i in list1]
                test_idx = [int(i) for i in list2]

                self.train_list = train_idx
                self.test_list = test_idx
                print(self.train_list,self.test_list)

            self.es.initialize()

            if training_type == 'attn':

                train_loader, test_loader = self.generate_dataloader(IMAGE_IDX_PATH, IMAGE_PATH, NORM_IMAGE_PATH,
                                                                     IMAGE_LIST,
                                                                     CLINICAL_PATH, RESULT_PATH, AUG_TIME, BATCH_SIZE, NUM_WORKERS,
                                                                     PIN_MEMORY,attnmodel)

                for clinical_idx in range(clinical_num):
                    self.attn_training(model=copy.deepcopy(attnmodel),
                                          EPOCH=EPOCH,
                                          train_value_path=os.path.join(value_path, str(idx),
                                                                        'train' + str(clinical_idx) + '.txt'),
                                          test_value_path=os.path.join(value_path, str(idx),
                                                                       'test' + str(clinical_idx) + '.txt'),
                                          train_result_path=os.path.join(value_path, str(idx), 'train.xlsx'),
                                          test_result_path=os.path.join(value_path, str(idx), 'test.xlsx'),
                                          for_idx=idx,
                                          image_path=os.path.join(value_path, str(idx), 'img'),
                                          model_path=os.path.join(value_path, str(idx),
                                                                  'abl_model' + str(clinical_idx) + '.pkl'),

                                        train_loader=train_loader,
                                        test_loader=test_loader,
                                          clinical_idx=clinical_idx
                                          )

            elif training_type=='hic' or training_type=='fc':
                if training_type=='hic':

                    train_loader, test_loader = self.generate_dataloader(IMAGE_IDX_PATH, IMAGE_PATH, NORM_IMAGE_PATH,
                                                                     IMAGE_LIST,
                                                                     CLINICAL_PATH, RESULT_PATH, AUG_TIME, BATCH_SIZE, NUM_WORKERS,
                                                                     PIN_MEMORY,attnmodel, include_list=include_list)
                else:
                    train_loader, test_loader = self.generate_dataloader(IMAGE_IDX_PATH, IMAGE_PATH, NORM_IMAGE_PATH,
                                                                     IMAGE_LIST,
                                                                     CLINICAL_PATH, RESULT_PATH, 1, BATCH_SIZE, NUM_WORKERS,
                                                                     PIN_MEMORY,attnmodel, include_list=include_list)

                self.hic_training(
                              model=model,
                              EPOCH=EPOCH,
                              train_value_path=os.path.join(value_path, str(idx), training_type + '_train.txt'),
                              test_value_path=os.path.join(value_path, str(idx), training_type + '_test.txt'),
                              model_path=os.path.join(value_path, str(idx), training_type + '_model.pkl'),
                              training_type=training_type,
                              train_best_path=os.path.join(value_path, str(idx),
                                                           'best_' + training_type + '_train.txt'),
                              test_best_path=os.path.join(value_path, str(idx), 'best_' + training_type + '_test.txt'),
                              model_best_path=os.path.join(value_path, str(idx),
                                                           'best_' + training_type + '_model.pkl'),
                              train_loader=train_loader,
                              test_loader=test_loader
                              )

            else:

                train_loader, test_loader = self.generate_dataloader(IMAGE_IDX_PATH, IMAGE_PATH, NORM_IMAGE_PATH,
                                                                     IMAGE_LIST, CLINICAL_PATH, RESULT_PATH, AUG_TIME,
                                                                     BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, attnmodel, True,
                                                                     os.path.join(value_path,str(idx)),clinical_num,
                                                                     include_list)
                

                self.training(attn_model=attnmodel,
                                 model=model,
                                 EPOCH=EPOCH,
                                 train_value_path=os.path.join(value_path, str(idx), training_type+'_train.txt'),
                                 test_value_path=os.path.join(value_path, str(idx), training_type+'_test.txt'),
                                 train_result_path=os.path.join(value_path, str(idx), training_type+'_train.xlsx'),
                                 test_result_path=os.path.join(value_path, str(idx), training_type+'_test.xlsx'),
                                 for_idx=idx,
                                 image_path=os.path.join(value_path, str(idx), 'img'),
                                 model_path=os.path.join(value_path, str(idx), training_type+'_model.pkl'),
                                 dir_path=os.path.join(value_path, str(idx)),
                                 training_type=training_type,
                                 train_best_path=os.path.join(value_path, str(idx), 'best_'+training_type+'_train.txt'),
                                 test_best_path=os.path.join(value_path, str(idx), 'best_'+training_type+'_test.txt'),
                                 model_best_path=os.path.join(value_path, str(idx), 'best_'+training_type+'_model.pkl'),
                                train_loader=train_loader,
                                test_loader=test_loader
                                 )




