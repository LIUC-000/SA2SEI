# import torchvision.models as models
# import torch
from mlp_head import MLPHead
from torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv, ComplexConv_trans

class Encoder_and_projection(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder_and_projection, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=4,stride=2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=4,stride=2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(896, 1024)
        self.projetion = MLPHead(in_channels=1024, **kwargs['projection_head'])

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)

        x = self.flatten(x)
        x = self.fc(x)
        embedding = F.relu(x)
        project_out = self.projetion(embedding)
        return embedding, project_out
