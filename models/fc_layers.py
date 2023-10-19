import torch.nn as nn
import torch
import torch.nn.functional as F


class ThreeLayersFC(nn.Module):
    def __init__(self,input_num):
        super().__init__()
        self.fc1 = nn.Linear(input_num,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,1)
    def  forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        #print(out.size())

        return out

class TwoLayersFC(nn.Module):
    def __init__(self, input_num):
        super().__init__()
        self.fc1 = nn.Linear(input_num, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        # print(out.size())

        return out
