
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(1024,10)

    def forward(self,x):
        # x = self.dropout(x)
        x = self.linear1(x)
        return x