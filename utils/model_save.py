import torch

def dataparalell_save(model,path):
    torch.save(model.module.state_dict(), path)