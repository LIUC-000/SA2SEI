import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
print(sys.path)
sys.path.insert(0, './models')
import torch
import yaml
import random
import numpy as np
from models.mlp_head import MLPHead
from models.encoder_and_projection import Encoder_and_projection
from trainer import BYOLTrainer
from get_dataset import PreTrainDataset_prepared
from torch.utils.data import TensorDataset, DataLoader

print(torch.__version__)

RANDOM_SEED = 300 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
set_seed(RANDOM_SEED)


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    X_train_ul, Y_train_ul = PreTrainDataset_prepared()
    train_dataset = TensorDataset(torch.Tensor(X_train_ul), torch.Tensor(Y_train_ul))

    # online network
    online_network = Encoder_and_projection(**config['network']).to(device)

    # predictor network
    predictor = MLPHead(in_channels=config['network']['projection_head']['projection_size'],
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = Encoder_and_projection(**config['network']).to(device)

    optimizer = torch.optim.Adam(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
