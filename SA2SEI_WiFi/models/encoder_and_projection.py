# import torchvision.models as models
# import torch
from mlp_head import MLPHead
from torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv, ComplexConv_trans

class Encoder_and_projection(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder_and_projection, self).__init__()
        self.conv1 = ComplexConv(in_channels=1, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)
        self.maxpool6 = nn.MaxPool1d(kernel_size=2)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)
        self.maxpool7 = nn.MaxPool1d(kernel_size=2)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)
        self.maxpool8 = nn.MaxPool1d(kernel_size=2)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4, stride=1)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)
        self.maxpool9 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(1024)
        self.projetion = MLPHead(in_channels=1024, **kwargs['projection_head'])

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)

        x = self.conv7(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)

        x = self.conv8(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)

        x = self.conv9(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)

        x = self.flatten(x)

        embedding = self.linear1(x)
        project_out = self.projetion(embedding)
        return embedding, project_out
