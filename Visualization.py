import torch
import sys
print(sys.path)
sys.path.insert(0, './models')
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from get_dataset import *
import os
import sklearn.metrics as sm
from sklearn import manifold
from models.encoder_and_projection import Encoder_and_projection
import yaml

def visualize_data(data, labels, title, num_clusters):  # feature visualization
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    data_tsne = tsne.fit_transform(data)
    fig = plt.figure(figsize=(6.3, 5))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], lw=0, s=10, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=600)

def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            #target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                #target = target.to(device)
            output = model(data)
            feature_map[len(feature_map):len(output[0])-1] = output[0].tolist()
            target_output[len(target_output):len(target)-1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output

def main():
    X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(62, range(10, 16), 100)

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    # online network
    model = Encoder_and_projection(**config['network'])

    checkpoints_folder = os.path.join('./runs',
                                      f"seed300_pretrain_62ft_class_in_0-9",
                                      'checkpoints')

    # load pre-trained parameters
    load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                             map_location='cpu')

    model.load_state_dict(load_params['online_network_state_dict'])

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    X_test_embedding_feature_map, target = obtain_embedding_feature_map(model, test_dataloader)

    visualize_data(X_test_embedding_feature_map, target.astype('int64'), "feature_visual", 6)
    print(sm.silhouette_score(X_test_embedding_feature_map, target, sample_size=len(X_test_embedding_feature_map), metric='euclidean'))

if __name__ == "__main__":
    main()
