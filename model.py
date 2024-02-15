import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

class model(nn.Module):
    def __init__(self, attribute_num, conditional_flg, bottle):
        super(model, self).__init__()

        self.attribute_num = attribute_num
        self.conditional_flg = conditional_flg #属性を追加するかしないか
        self.bottle = bottle

        
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, self.bottle)

        self.conditional_fc = nn.Linear(self.bottle + attribute_num, 7)
        self.non_conditional_fc = nn.Linear(self.bottle, 7)

        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x, attribute):

        if self.conditional_flg==1:
            x = self.resnet18(x)
            x = torch.cat((x, attribute), 1)
            x = self.conditional_fc(x)
            x = self.sigmoid(x)
        else:
            x = self.resnet18(x)
            x = self.non_conditional_fc(x)
            x = self.sigmoid(x)

        return x