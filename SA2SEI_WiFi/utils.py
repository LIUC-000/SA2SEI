import os
from shutil import copyfile
import torch.nn.functional as F
import torch
import torch.nn as nn
from AdvAugment import AdvAug

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

class Data_augment(nn.Module):
    def __init__(self, rotate=False, flip=False, rotate_and_flip=False, mask=False, awgn=False, add_noise=False, slice=False, isAdvAug=False):
        super(Data_augment, self).__init__()

        self.rotate = rotate
        self.rotate_angle = [0,90,180,270]

        self.flip = flip

        self.rotate_and_flip = rotate_and_flip

        self.mask = mask

        self.awgn = awgn
        self.noise_snr = [10, 20]  # 10~20

        self.add_noise = add_noise
        self.mean = 0
        self.std = 0.1

        self.slice = slice
        self.slice_len = 2400

        self.isAdvAug = isAdvAug
        self.n_power = 1
        self.XI = 0.01
        self.epsilon = 1.0

    def rotation_2d(self, x, ang):
        x_aug = torch.zeros(x.shape)
        if ang == 0:
            x_aug = x
        elif ang == 90:
            x_aug[0, :] = -x[1, :]
            x_aug[1, :] = x[0, :]
        elif ang == 180:
            x_aug = -x
        elif ang == 270:
            x_aug[0, :] = x[1, :]
            x_aug[1, :] = -x[0, :]
        else:
            print("Wrong input for rotation!")
        return x_aug

    def get_normalized_vector(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def Rotate(self, x):
        x_rotate = torch.zeros(x.shape)
        for i in range(x.shape[0]):
            ang = self.rotate_angle[torch.randint(len(self.rotate_angle), [1])]
            x_rotate[i, :, :] = self.rotation_2d(x[i, :, :], ang)
        return x_rotate.cuda()

    def Flip(self, x):
        mul = [-1, 1]
        for i in range(x.shape[0]):
            I_mul = mul[torch.randint(len(mul), [1])]
            Q_mul = mul[torch.randint(len(mul), [1])]
            x[i, 0, :] = I_mul * x[i, 0, :]
            x[i, 1, :] = Q_mul * x[i, 1, :]
        return x

    def Rotate_and_Flip(self, x):
        choice_list = ['Rotate', 'Flip', 'Flip']
        rotate_angle = [90, 270]
        mul = [-1, 1]
        for i in range(x.shape[0]):
            choice = choice_list[torch.randint(len(choice_list), [1])]
            if choice == 'Rotate':
                ang = rotate_angle[torch.randint(len(rotate_angle), [1])]
                x[i, :, :] = self.rotation_2d(x[i, :, :], ang)
            else:
                I_mul = mul[torch.randint(len(mul), [1])]
                Q_mul = mul[torch.randint(len(mul), [1])]
                x[i, 0, :] = I_mul * x[i, 0, :]
                x[i, 1, :] = Q_mul * x[i, 1, :]
        return x

    def Mask(self, x, mask_len=800):
        start = torch.randint(0, x.shape[2] - mask_len, [1])
        end = start + mask_len
        x[:, :, start:end] = 0
        return x

    def Awgn(self, x, snr):
        snr = torch.randint(snr[0], snr[1], [1])
        snr = 10 ** (snr / 10.0)
        x_awgn = torch.zeros((x.shape[0], x.shape[1]))
        # real
        x_real = x[:, 0]
        xpower_real = torch.sum(x_real ** 2) / len(x_real)
        npower_real = xpower_real / snr
        noise_real = torch.randn(len(x_real)) * torch.sqrt(npower_real)
        x_awgn[:, 0] = x_real + noise_real
        # imag
        x_imag = x[:, 1]
        xpower_imag = torch.sum(x_imag ** 2) / len(x_imag)
        npower_imag = xpower_imag / snr
        noise_imag = torch.randn(len(x_imag)) * torch.sqrt(npower_imag)
        x_awgn[:, 1] = x_imag + noise_imag
        return x_awgn

    def Add_noise(self, x):
        d = torch.normal(mean=self.mean, std=self.std, size=(x.shape[0], x.shape[1], x.shape[2]))
        return x + d

    def Slice(self, x):
        start = torch.randint(0, x.shape[2] - self.slice_len, [1])
        end = start + self.slice_len
        return x[:, :, start:end]

    def forward(self, x, online_network=None, mask_len=800):
        x = x.cuda()
        if self.rotate:
            x = self.Rotate(x)
        if self.flip:
            x = self.Flip(x)
        if self.rotate_and_flip:
            x = self.Rotate_and_Flip(x)
        if self.mask:
            x = self.Mask(x, mask_len)
        if self.awgn:
            x = self.Awgn(x, self.noise_snr)
        if self.add_noise:
            x = self.Add_noise(x)
        if self.slice:
            x = self.Slice(x)
        if self.isAdvAug:
            aug_gen = AdvAug(online_network, self.n_power, self.XI, self.epsilon)
            x = aug_gen(x)
        return x.detach()