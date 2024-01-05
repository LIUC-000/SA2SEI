import os
import sys
print(sys.path)
sys.path.insert(0, 'models')
import torch
import yaml
from models.encoder_and_projection import Encoder_and_projection
from models.classifier import Classifier
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from get_dataset import FineTuneDataset_prepared
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

os.environ['CUDA_VISIBLE_DEVICES']='0'

def train(online_network, classifier, loss_nll, train_dataloader, optim_online_network, optimizer_classifier, scheduler_online_network, scheduler_classifier, epoch, device, writer):
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

        features = online_network(data)[0]
        output = F.log_softmax(classifier(features), dim=1)
        nll_loss_batch = loss_nll(output, target)
        nll_loss_batch.backward()

        optim_online_network.step()
        optimizer_classifier.step()
        # scheduler_online_network.step()
        # scheduler_classifier.step()

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
    online_network.eval()
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
    return test_loss

def test(online_network, classifier, test_dataloader, device):
    online_network.eval()
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

def train_and_test(online_network, classifier, loss_nll, train_dataloader, val_dataloader, optim_online_network, optim_classifier, scheduler_online_network, scheduler_classifier, epochs, save_path_online_network, save_path_classifier, device, writer):
    current_min_test_loss = 10000000000
    for epoch in range(1, epochs + 1):
        train(online_network, classifier, loss_nll, train_dataloader, optim_online_network, optim_classifier, scheduler_online_network, scheduler_classifier, epoch, device, writer)
        validation_loss = evaluate(online_network, classifier, loss_nll, val_dataloader, epoch, device, writer)
        if validation_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, validation_loss))
            current_min_test_loss = validation_loss
            torch.save(online_network, save_path_online_network)
            torch.save(classifier, save_path_classifier)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

def run(train_dataloader, val_dataloader, test_dataloader, epochs, save_path_online_network, save_path_classifier, device, writer, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    checkpoints_folder = os.path.join('runs',
                                      f"seed300_pretrain_class_in_{config['trainer']['class_start']}-{config['trainer']['class_end']}",
                                      'checkpoints')
    online_network = torch.load(os.path.join(checkpoints_folder, 'model_best.pth'))
    classifier = Classifier()

    loss_nll = nn.NLLLoss()
    if torch.cuda.is_available():
        online_network = online_network.to(device)
        classifier = classifier.to(device)
        loss_nll = loss_nll.to(device)

    optim_online_network = torch.optim.Adam(online_network.parameters(), lr=config['finetune']['lr'])
    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=config['finetune']['lr'])

    scheduler_online_network = CosineAnnealingLR(optim_online_network, T_max=20)
    scheduler_classifier = CosineAnnealingLR(optim_classifier, T_max=20)

    train_and_test(online_network, classifier, loss_nll, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optim_online_network=optim_online_network, optim_classifier=optim_classifier, scheduler_online_network=scheduler_online_network, scheduler_classifier=scheduler_classifier, epochs=epochs, save_path_online_network=save_path_online_network, save_path_classifier=save_path_classifier, device=device, writer=writer)
    print("Test_result:")
    online_network = torch.load(save_path_online_network)
    classifier = torch.load(save_path_classifier)
    test_acc = test(online_network, classifier, test_dataloader, device)
    return test_acc

def main():
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_ft = config['finetune']

    device = torch.device("cuda:0")

    test_acc_all = []

    for i in range(config['iteration']):
        print(f"iteration: {i}--------------------------------------------------------")
        set_seed(i)
        writer = SummaryWriter(f"./log_finetune/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot")

        save_path_classifier = f"./model_weight/classifier_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"
        save_path_online_network = f"./model_weight/online_network_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth"

        X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared()
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=30)

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=config_ft['batch_size'], shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_dataloader = DataLoader(val_dataset, batch_size=config_ft['batch_size'], shuffle=True)

        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'], shuffle=True)

        # train
        test_acc = run(train_dataloader, val_dataloader, test_dataloader, epochs=config_ft['epochs'], save_path_online_network=save_path_online_network, save_path_classifier=save_path_classifier, device=device, writer=writer, config=config)
        test_acc_all.append(test_acc)
        writer.close()

    df = pd.DataFrame(test_acc_all)
    df.to_excel(f"test_result/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot.xlsx")

if __name__ == '__main__':
   main()
