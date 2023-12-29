import json
import matplotlib.pyplot as plt

from medcam import medcam
import nibabel as nib
import numpy as np
import os
import torch.nn.functional as F
import torch

class Ascender:
    def __init__(self,layer=None):
        self.layer = layer
        pass
    def get_grad_cam(self,model,sample,layer=None):
        model.eval()
        if self.layer is not None:
            layer = self.layer
        model = medcam.inject(model, layer = layer)
        model(sample)
        heatmap = model.get_attention_map()
        return heatmap

    def up_sample_array(self,arr, aim_shape):
        ori_shape = arr.shape
        if len(ori_shape) == 5:
            arr = torch.Tensor(arr)
        elif len(ori_shape) == 3:
            arr = torch.Tensor(arr.reshape(1, 1, *ori_shape))
        arr = F.interpolate(arr, size=aim_shape, mode='nearest')
        arr = arr.numpy()
        return arr





