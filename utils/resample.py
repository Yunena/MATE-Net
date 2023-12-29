import torch
import torch.nn.functional as F


def re_sample(sample,aim=(8,32,32),stype='tensor'):
    if not stype=='tensor':
        sample = torch.Tensor(sample)
    if len(sample.size())==3:
        sample = sample.unsqueeze(0)
    if len(sample.size())==4:
        sample = sample.unsqueeze(0)
    if len(sample.size())==5:
        sampled_case = F.interpolate(sample,size=aim,mode='nearest')
        return sampled_case



