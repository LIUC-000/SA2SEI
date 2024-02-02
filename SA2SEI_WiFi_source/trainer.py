import os
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy
from utils import _create_model_training_folder, Data_augment

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter(f"runs/seed300_pretrain_{params['ft']}ft_class_in_{params['class_start']}-{params['class_end']}")
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        Augment = Data_augment(rotate=False, flip=False, rotate_and_flip=True, mask=False, awgn=False, add_noise=False, slice=False, isAdvAug=True)

        loss_min = 10000000
        for epoch_counter in range(self.max_epochs):
            print(f'Epoch={epoch_counter}')
            loss_epoch = 0
            for batch_view, _ in train_loader:
                batch_view = batch_view.to(self.device)
                batch_view_1 = Augment(batch_view, online_network=copy.deepcopy(self.online_network))
                batch_view_2 = Augment(batch_view, online_network=copy.deepcopy(self.online_network))

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch += loss.item()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1

            loss_epoch /= len(train_loader)
            if loss_epoch <= loss_min:
                self.save_model(os.path.join(model_checkpoints_folder, 'model_best.pth'))
                loss_min = loss_epoch
            self.writer.add_scalar('loss_epoch', loss_epoch, global_step=epoch_counter)
            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1)[1])
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2)[1])

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)[1]
            targets_to_view_1 = self.target_network(batch_view_2)[1]

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
