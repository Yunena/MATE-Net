import torchio as tio
import os
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TARGET_SIZE = (256, 256, 32)
croporpadtrans = tio.CropOrPad(TARGET_SIZE)
def sample_one_img_load(idx,image_path,image_name):
    image_path = os.path.join(image_path, str(idx))
    image_file = os.path.join(image_path, image_name)
    image = croporpadtrans(tio.ScalarImage(image_file)).data.numpy()
    image = np.transpose(image, (0, 3, 1, 2))
    return image[0]

def sample_load(idx,image_path,clinical_path,result_path,
              image_list=['AP.nii', 'AP2.nii', 'CTP0.nii', 'CBV.nii.gz', 'CBF.nii.gz', 'Tmax.nii.gz', 'mttv.nii.gz',\
                          'core_label.nii.gz', 'deficit_label.nii.gz'],return_type='tensor'):
    image_path = os.path.join(image_path,str(idx))
    nplist = []
    for i, image_name in enumerate(image_list):
        image_file = os.path.join(image_path,image_name)
        image = croporpadtrans(tio.ScalarImage(image_file)).data.numpy()
        image = np.transpose(image, (0, 3, 1, 2))
        nplist.append(image[0])
        del (image)

    clinical_data = pd.read_excel(clinical_path,header=0,index_col=0)
    clinical = clinical_data.loc[idx]
    result = pd.read_excel(result_path,header=0,index_col=0).loc[idx]
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(scaler.fit_transform(clinical_data),index=clinical_data.index).loc[idx]

    new_arr = []
    for i,col_name in enumerate(clinical_data.columns):
        arr = np.array([clinical_data.iloc[:,i]])
        arr = np.transpose(arr,(1,0))
        if len(np.unique(arr))>2:
            zscore = StandardScaler()
            arr = zscore.fit_transform(arr)
        else:
            arr[arr==1] = np.mean(arr)
        new_arr.append(arr)

    new_arr = np.concatenate(new_arr,axis=1)

    zscored = pd.DataFrame(new_arr,index = clinical_data.index, columns=clinical_data.columns).loc[idx]



    if return_type=='tensor':
        sample = {"image": torch.Tensor(nplist).unsqueeze(0), \
              "clinical": torch.Tensor(clinical).unsqueeze(0), \
              "result": torch.Tensor(result),\
              "normalized":torch.Tensor(normalized),\
              "zscored":torch.Tensor(zscored)
              }
    elif return_type=='array':
        sample = {"image": np.array([nplist]), \
              "clinical": np.array([clinical]), \
              "result": np.array([result]),\
              "normalized":np.array(normalized),\
              "zscored":np.array(zscored)
              }


    return sample