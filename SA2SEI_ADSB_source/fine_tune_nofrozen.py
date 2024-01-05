import os
import sys
print(sys.path)
sys.path.insert(0, './models')
import torch
import yaml
from models.encoder_and_projection import Encoder_and_projection
from models.classifier import Classifier
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from get_dataset import FineTuneDataset_prepared
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random
from math import pi
from math import cos
from math import floor

# RANDOM_SEED = 300 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


os.environ['CUDA_VISIBLE_DEVICES']='1'

parser = argparse.ArgumentParser(description='PyTorch Complex_test Training')
parser.add_argument('--lr_encoder', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
parser.add_argument('--lr_classifier', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
args = parser.parse_args(args=[])

def train(online_network, classifier, loss_nll, train_dataloader, optim_online_network, optimizer_classifier, epoch, device, writer):
    online_network.train()  # 启动训练, 允许更新模型参数
    classifier.train()
    correct = 0
    nll_loss = 0
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        optim_online_network.zero_grad()
        optimizer_classifier.zero_grad()

        # 分类损失反向生成encoder和classifier的梯度
        features = online_network(data)[0]
        output = F.log_softmax(classifier(features), dim=1)
        nll_loss_batch = loss_nll(output, target)
        nll_loss_batch.backward()

        optim_online_network.step()
        optimizer_classifier.step()

        nll_loss += nll_loss_batch.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()  # 求pred和target中对应位置元素相等的个数

    nll_loss /= len(train_dataloader)

    print('Train Epoch: {} \tClass_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        nll_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', nll_loss, epoch)

def evaluate(online_network, classifier, loss_nll, val_dataloader, epoch, device, writer):
    online_network.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = classifier(online_network(data)[0])
            output = F.log_softmax(output, dim=1)
            test_loss += loss_nll(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_dataloader)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )
    writer.add_scalar('Accuracy/val', 100.0 * correct / len(val_dataloader.dataset), epoch)
    writer.add_scalar('Loss/val', test_loss, epoch)

def test(online_network, classifier, test_dataloader, device):
    online_network.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
    test_loss = 0
    correct = 0
    loss = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)
            output = classifier(online_network(data)[0])
            output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset)

# def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max, lrate_min):
#     epochs_per_cycle = floor(n_epochs / n_cycles)
#     cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
#     b = (lrate_max - lrate_min) / 2
#     return cos(cos_inner)*b + b + lrate_min
#
# def adjust_learning_rate(optim_classifier, epochs ,epoch, cycles):
#     """decrease the learning rate"""
#     lr_min = 0.0001
#     lr_classifier = 0.001
#     lr_classifier_new = cosine_annealing(epoch-1, epochs, cycles, lr_classifier, lr_min)
#
#     for param_group in optim_classifier.param_groups:
#         param_group['lr'] = lr_classifier_new

def train_and_test(online_network, classifier, loss_nll, train_dataloader, val_dataloader, optim_online_network, optim_classifier, epochs, save_path_online_network, save_path_classifier, device, writer):
    # cycles = 1  # 余弦退火学习率周期轮次
    for epoch in range(1, epochs + 1):
        # adjust_learning_rate(optim_classifier, epochs, epoch, cycles)
        train(online_network, classifier, loss_nll, train_dataloader, optim_online_network, optim_classifier, epoch, device, writer)
        # evaluate(online_network, classifier, loss_nll, val_dataloader, epoch, device, writer)
    torch.save(online_network, save_path_online_network)
    torch.save(classifier, save_path_classifier)

def run(train_dataloader, val_dataloader, test_dataloader, epochs, save_path_online_network, save_path_classifier, device, writer, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    # online network
    online_network = Encoder_and_projection(**config['network']).to(device)

    # load pre-trained model if defined
    try:
        checkpoints_folder = os.path.join('./runs', f"seed300_pretrain_class_in_{config['trainer']['class_start']}-{config['trainer']['class_end']}", 'checkpoints')

        # load pre-trained parameters
        load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                 map_location=torch.device(torch.device(device)))

        online_network.load_state_dict(load_params['online_network_state_dict'])

    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")

    classifier = Classifier()

    # for param in online_network.parameters():
    #     param.requires_grad = False  # not update by gradient

    if torch.cuda.is_available():
        online_network = online_network.to(device)
        classifier = classifier.to(device)

    loss_nll = nn.NLLLoss()
    if torch.cuda.is_available():
        loss_nll = loss_nll.to(device)

    optim_online_network = torch.optim.Adam(online_network.parameters(), lr=args.lr_classifier, weight_decay=0.0001)
    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=args.lr_classifier, weight_decay=0.0001)

    train_and_test(online_network, classifier, loss_nll, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optim_online_network=optim_online_network, optim_classifier=optim_classifier, epochs=epochs, save_path_online_network=save_path_online_network, save_path_classifier=save_path_classifier, device=device, writer=writer)
    print("Test_result:")
    online_network = torch.load(save_path_online_network)
    classifier = torch.load(save_path_classifier)
    test_acc = test(online_network, classifier, test_dataloader, device)
    return test_acc

def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_ft = config['finetune']

    device = torch.device("cuda:0")

    test_acc_all = []

    for i in range(config['iteration']):
        print(f"iteration: {i}--------------------------------------------------------")
        set_seed(i)
        writer = SummaryWriter(f"./log_finetune/nofrozen_and_onelinear/pt_{config['trainer']['class_start']}-{config['trainer']['class_end']}_ft_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot")

        save_path_classifier = f"./model_weight/nofrozen_and_onelinear/classifier_pt_{config['trainer']['class_start']}-{config['trainer']['class_end']}_ft_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        save_path_online_network = f"./model_weight/nofrozen_and_onelinear/online_network_pt_{config['trainer']['class_start']}-{config['trainer']['class_end']}_ft_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"

        X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(config_ft['k_shot'])

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=config_ft['batch_size'], shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        val_dataloader = DataLoader(val_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)

        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)

        # train
        test_acc = run(train_dataloader, val_dataloader, test_dataloader, epochs=config_ft['epochs'], save_path_online_network=save_path_online_network, save_path_classifier=save_path_classifier, device=device, writer=writer, config=config)
        test_acc_all.append(test_acc)
        writer.close()

    df = pd.DataFrame(test_acc_all)
    df.to_excel(f"test_result/nofrozen_and_onelinear/pt_{config['trainer']['class_start']}-{config['trainer']['class_end']}_ft_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot.xlsx")

if __name__ == '__main__':
   main()