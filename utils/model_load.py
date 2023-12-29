import torch

def load_model(model,path,device='cpu'):
    device = torch.device(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def load_partmodel(model,path,device = 'cpu'):
    model.eval()
    keys = model.state_dict().keys()

    save_model = torch.load(path, map_location=device)

    state_dict = {k: v for k, v in save_model.items() if k in keys}
    model_dict = model.state_dict()
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model

