import torch
import torchmetrics
import numpy as np
import sklearn.metrics as skm


def reg_evaluation(torch_list,target,thred=3,max_value = 6, device='gpu'):
    torch_np = torch_list.detach().numpy()

    bp_np = torch_np.copy()
    target_np = target.numpy().astype('int')
    #print('target:',target_np)
    #print(torch_np,target_np)
    pred_np = np.round(torch_np)
    #print('float pred:',np.round(torch_np))
    pred = torch.from_numpy(np.round(torch_np).astype('int'))
    pred = pred.t()
    if(torch.min(pred)<0):
        for p in pred[0]:
            print(p)
            if p<0:
                p=0
    pred = pred.t()
    target = torch.from_numpy(target_np.copy())
    for i in range(len(torch_np)):
        bp_np[i] = 0 if pred_np[i]<thred else 1
        target_np[i] = 0 if target_np[i]<thred else 1
    binary_pred = torch.from_numpy((torch_np / max_value))
    #binary_pred = torch.from_numpy(bp_np)
    bp = torch.from_numpy(bp_np.astype('float'))
    #print(binary_pred)
    binary_target = torch.from_numpy(target_np)
    bt = torch.from_numpy(target_np)

    r2 = torchmetrics.R2Score()
    acc = torchmetrics.Accuracy('multiclass',num_classes=int(max_value+1))
    biacc = torchmetrics.Accuracy('binary')
    auc = torchmetrics.AUROC(task = 'binary')
    roc = torchmetrics.ROC(task = 'binary')
    pre = torchmetrics.Precision(task = 'binary')
    rec = torchmetrics.Recall(task = 'binary')
    spe = torchmetrics.Specificity(task = 'binary')
    f1 = torchmetrics.F1Score(task = 'binary')
    cm = torchmetrics.ConfusionMatrix(task = 'binary',num_classes=2)



    acc_r = acc(pred,target)
    r2_r = r2(torch_list,target) if len(torch_list) >1 else torch.Tensor([0.00])
    biacc_r = biacc(bp,bt)
    auc_r = auc(binary_pred,binary_target)
    fpr, tpr, _ = roc(binary_pred, binary_target)
    pre_r = pre(bp, bt)
    rec_r = rec(bp, bt)
    spe_r = spe(bp, bt)
    cm_r = cm(bp, bt)
    #print(cm_r)
    npv_r = cm_r[0][0]/torch.sum(cm_r,axis=0)[0] if cm_r[0][0]!=0 else cm_r[0][0]
    #print(npv_r)
    f1_r = f1(bp,bt)

    if device!='cpu':
        acc_r = acc_r.data.cpu().numpy()
        r2_r = r2_r.data.cpu().numpy()
        biacc_r = biacc_r.data.cpu().numpy()
        auc_r = auc_r.data.cpu().numpy()
        #fpr, tpr, _ = roc(binary_pred_float, bt)
        fpr = fpr.data.cpu().numpy().tolist()
        tpr = tpr.data.cpu().numpy().tolist()
        pre_r = pre_r.data.cpu().numpy()
        rec_r = rec_r.data.cpu().numpy()
        spe_r = spe_r.data.cpu().numpy()
        #cm_r = cm_r.data.cpu().numpy()
        # print(cm_r)
        npv_r = npv_r.data.cpu().numpy()
        # print(npv_r)
        f1_r = f1_r.data.cpu().numpy()


    #F1 = torchmetrics.F1()


    return acc_r, r2_r, biacc_r, auc_r, fpr, tpr, pre_r, rec_r, spe_r, npv_r,f1_r

def prc_evaluation(torch_list,target,thred=3,max_value = 6, device='gpu'):
    #print('pred:',torch_list.t())
    #torch_np = torch_list.detach().numpy().astype('int')
    #if device == 'gpu':
    torch_np = torch_list.detach().numpy()

    bp_np = torch_np.copy()
    target_np = target.numpy().astype('int')
    #print('target:',target_np)
    #print(torch_np,target_np)
    pred_np = np.round(torch_np)
    #print('float pred:',np.round(torch_np))
    pred = torch.from_numpy(np.round(torch_np).astype('int'))
    pred = pred.t()
    if(torch.min(pred)<0):
        for p in pred[0]:
            print(p)
            if p<0:
                p=0
    pred = pred.t()
    target = torch.from_numpy(target_np.copy())
    for i in range(len(torch_np)):
        bp_np[i] = 0 if pred_np[i]<thred else 1
        target_np[i] = 0 if target_np[i]<thred else 1
    binary_pred = torch.from_numpy((torch_np / max_value))
    #binary_pred = torch.from_numpy(bp_np)
    bp = torch.from_numpy(bp_np.astype('float'))
    #print(binary_pred)
    binary_target = torch.from_numpy(target_np)
    bt = torch.from_numpy(target_np)

    prc = torchmetrics.PrecisionRecallCurve(task = 'binary')

    pre, rec, _ = prc(binary_pred,binary_target)

    #print(pred,target)
    #print('target:',target.t())
    #print(torch_list.t())
    #print(pred.t())

    if device!='cpu':

        #fpr, tpr, _ = roc(binary_pred_float, bt)
        pre = pre.data.cpu().numpy().tolist()
        rec = rec.data.cpu().numpy().tolist()


    #F1 = torchmetrics.F1()


    return pre,rec

def reg_disc_eval(max_value,thred,torch_list,target):
    #torch_np = torch_list.detach().numpy().astype('int')
    torch_np = torch_list.detach().numpy()
    bp_np = torch_np.copy()
    target_np = target.numpy().astype('int')
    #print('target:',target_np)
    #print(torch_np,target_np)
    for i in range(len(torch_np)):
        if torch_np[i]>6:
            torch_np[i]=6
        elif torch_np[i]<0:
            torch_np[i]=0

    pred_np = np.round(torch_np)
    #print('float pred:',np.round(torch_np))
    pred = torch.from_numpy(np.round(torch_np).astype('int'))
    target = torch.from_numpy(target_np.copy())
    for i in range(len(torch_np)):
        bp_np[i] = 0 if pred_np[i]<thred else 1
        target_np[i] = 0 if target_np[i]<thred else 1
    #binary_pred = torch.from_numpy((torch_np/(max_value)))
    binary_pred = torch.from_numpy(bp_np)
    bp = torch.from_numpy(bp_np.astype('int'))
    #print(binary_pred)
    binary_target = torch.from_numpy(target_np)
    bt = torch.from_numpy(target_np)
    acc = torchmetrics.Accuracy()
    auc = torchmetrics.AUROC(pos_label=1)
    roc = torchmetrics.ROC(pos_label=1)
    pre = torchmetrics.Precision()
    rec = torchmetrics.Recall()
    spe = torchmetrics.Specificity()
    f1 = torchmetrics.F1()


    #print(pred,target)
    print('target:',target.t())
    print('pred:',pred.t())
    acc_r = acc(pred,target)
    biacc_r = acc(bp,bt)
    auc_r = auc(binary_pred,binary_target)
    fpr, tpr, _ = roc(binary_pred, binary_target)
    pre_r = pre(bp, bt)
    rec_r = rec(bp, bt)
    spe_r = spe(bp, bt)
    f1_r = f1(bp,bt)


    #F1 = torchmetrics.F1()

    return acc_r, biacc_r, auc_r, fpr, tpr, pre_r, rec_r, spe_r, f1_r

def muc_evaluation(torch_list,target,thred=3,device='gpu'):
    #pred = torch_list
    pred = torch.tensor([torch_list.argmax(dim=1).tolist().copy()])[0]
    binary_pred_int = torch.tensor([torch_list.argmax(dim=1).tolist().copy()])[0]
    binary_pred_float = torch.Tensor([torch_list.argmax(dim=1).tolist().copy()])[0]
    #print('pred:',binary_pred)
    bp_np_int = binary_pred_int.numpy()
    bp_np_float = binary_pred_float.numpy()
    target_np = target.t()[0].numpy().astype('int')
    bt_np = target_np.copy()
    target = torch.from_numpy(target_np.copy())
    #target = torch.nn.functional.one_hot(target, max_value+1)

    #print('target:',target)
    #print('pred:',pred)

    for i in range(len(bp_np_int)):
        bp_np_int[i] = 0 if bp_np_int[i]<thred else 1
        bp_np_float[i] = 0 if bp_np_float[i]<thred else 1
        bt_np[i] = 0 if bt_np[i]<thred else 1

    bp = torch.from_numpy(bp_np_int.copy())
    binary_pred_float = torch.from_numpy(bp_np_float.copy())
    bt = torch.from_numpy(bt_np.astype('int').copy())


    #print('target:',binary_target)
    #print('pred:',binary_pred)


    #print(pred)
    #print(target)
    acc = torchmetrics.Accuracy()
    r2 = torchmetrics.R2Score()
    biacc = torchmetrics.Accuracy()
    #roc = torchmetrics.ROC(num_classes=max_value+1)
    auc = torchmetrics.AUROC(pos_label=1)
    roc = torchmetrics.ROC(pos_label=1)
    pre = torchmetrics.Precision()
    rec = torchmetrics.Recall()
    spe = torchmetrics.Specificity()
    f1 = torchmetrics.F1()
    cm = torchmetrics.ConfusionMatrix(num_classes=2)
    #print(pred,target)



    acc_r = acc(pred,target)
    r2_r = r2(pred,target)
    biacc_r = biacc(bp,bt)
    auc_r = auc(binary_pred_float,bt)
    fpr, tpr, _ = roc(binary_pred_float, bt)
    pre_r = pre(binary_pred_float, bt)
    rec_r = rec(binary_pred_float, bt)
    spe_r = spe(binary_pred_float, bt)
    cm_r = cm(binary_pred_float, bt)
    #print(cm_r)
    npv_r = cm_r[0][0]/torch.sum(cm_r,axis=0)[0] if cm_r[0][0]!=0 else cm_r[0][0]
    #print(npv_r)
    f1_r = f1(binary_pred_float,bt)

    if device!='cpu':
        acc_r = acc_r.data.cpu().numpy()
        r2_r = r2_r.data.cpu().numpy()
        biacc_r = biacc_r.data.cpu().numpy()
        auc_r = auc_r.data.cpu().numpy()
        #fpr, tpr, _ = roc(binary_pred_float, bt)
        pre_r = pre_r.data.cpu().numpy()
        rec_r = rec_r.data.cpu().numpy()
        spe_r = spe_r.data.cpu().numpy()
        cm_r = cm_r.data.cpu().numpy()
        # print(cm_r)
        npv_r = npv_r.data.cpu().numpy()
        # print(npv_r)
        f1_r = f1_r.data.cpu().numpy()


    #F1 = torchmetrics.F1()

    return acc_r, r2_r, biacc_r, auc_r, fpr, tpr, pre_r, rec_r, spe_r, npv_r,f1_r


    #return acc_r, biacc_r, auc_r, fpr, tpr, pre_r, rec_r, spe_r,f1_r

def bic_evaluation(torch_list,target,device='gpu'):
    #print('pred:',torch_list.t())
    #print('output pred:',torch_list)
    bp = torch_list.t()[1].t()
    bt = target
    acc = torchmetrics.Accuracy()
    biacc = torchmetrics.Accuracy()
    auc = torchmetrics.AUROC(pos_label=1)
    roc = torchmetrics.ROC(pos_label=1)
    pre = torchmetrics.Precision()
    rec = torchmetrics.Recall()
    spe = torchmetrics.Specificity()
    f1 = torchmetrics.F1()
    cm = torchmetrics.ConfusionMatrix(num_classes=2)

    #print(pred,target)
    #print('pred:',bp.t())
    #print('target:',target.t())
    acc_r = acc(bp,bt)
    biacc_r = biacc(bp,bt)
    auc_r = auc(bp,bt)
    fpr, tpr, _ = roc(bp, bt)
    pre_r = pre(bp, bt)
    rec_r = rec(bp, bt)
    spe_r = spe(bp, bt)
    cm_r = cm(bp, bt)
    #print(cm_r)
    npv_r = cm_r[0][0]/torch.sum(cm_r,axis=0)[0] if cm_r[0][0]!=0 else cm_r[0][0]
    #print(npv_r)
    f1_r = f1(bp,bt)

    if device!='cpu':
        acc_r = acc_r.data.cpu().numpy()
        #r2_r = r2_r.data.cpu().numpy()
        biacc_r = biacc_r.data.cpu().numpy()
        auc_r = auc_r.data.cpu().numpy()
        #fpr, tpr, _ = roc(binary_pred_float, bt)
        fpr = fpr.data.cpu().numpy().tolist()
        tpr = tpr.data.cpu().numpy().tolist()
        pre_r = pre_r.data.cpu().numpy()
        rec_r = rec_r.data.cpu().numpy()
        spe_r = spe_r.data.cpu().numpy()
        #cm_r = cm_r.data.cpu().numpy()
        # print(cm_r)
        npv_r = npv_r.data.cpu().numpy()
        # print(npv_r)
        f1_r = f1_r.data.cpu().numpy()


    #F1 = torchmetrics.F1()

    return acc_r, biacc_r, auc_r, fpr, tpr, pre_r, rec_r, spe_r, npv_r,f1_r

def rf_evaluation(max_value,thred,pred,target):
    int_pred = np.round(pred).astype('int')
    bp_np = np.copy(pred)
    bt_np = np.copy(target)
    for i in range(len(pred)):
        bp_np[i] = 0 if bp_np[i]<thred else 1
        bt_np[i] = 0 if bt_np[i]<thred else 1
    binary_pred = torch.from_numpy((pred/(max_value)))

    acc_r = skm.accuracy_score(target,int_pred)
    r2_r = skm.r2_score(target,pred)
    biacc_r = skm.accuracy_score(bt_np,bp_np)
    fpr,tpr,_ = skm.roc_curve(bt_np,binary_pred)
    auc_r = skm.auc(fpr,tpr)
    pre_r = skm.precision_score(bt_np,bp_np)
    rec_r = skm.recall_score(bt_np,bp_np)
    cm_r = skm.confusion_matrix(bt_np,bp_np)
    spe_r = cm_r[0][0] / np.sum(cm_r, axis=1)[0] if cm_r[0][0] != 0 else cm_r[0][0]
    npv_r = cm_r[0][0] / np.sum(cm_r, axis=0)[0] if cm_r[0][0] != 0 else cm_r[0][0]
    f1_r = skm.f1_score(bt_np,bp_np)

    return acc_r, r2_r, biacc_r, auc_r, fpr, tpr, pre_r, rec_r, spe_r, npv_r,f1_r


