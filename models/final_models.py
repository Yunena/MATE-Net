import torch
import torch.nn as nn
import torch.nn.functional as F
from models import attn_module, pred_module
from models.fc_layers import ThreeLayersFC


class AttentionModule(nn.Module):
    def __init__(self,input_num=7):
        super(AttentionModule, self).__init__()
        self.model = attn_module.generate_model(18, n_input_channels=input_num)
        self.fc = ThreeLayersFC(512)

    def forward(self,x):
        out = self.model(x)
        out = self.fc(out)
        return out

class PredictionModule(nn.Module):
    def __init__(self,clinical_num=13,input_num=7,output_num=1,scale = 6):
        super(PredictionModule, self).__init__()
        self.im = pred_module.generate_model(18, clinical_num=clinical_num,n_input_channels=input_num)
        self.red = nn.Linear(512, 20)
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, output_num)
        self.features = None
        self.scale = scale
        self.output_num = output_num

    def forward(self, xi, sa=None):
        image_featrue = self.im(xi, sa)
        out = F.relu(self.red(image_featrue))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        if self.output_num == 1:
            return self.scale * F.sigmoid(out)
        else:
            return F.softmax(out, dim=1)
