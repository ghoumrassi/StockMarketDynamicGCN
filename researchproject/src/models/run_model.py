""" Environment for running models. """
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.models.models import ModelA_EvolveGCN_w_Dense
from src.data.datasets import CompanyStockGraphDataset

class Args:
    def __init__(self):
        self.node_feat_dim = 2
        self.layer_1_dim = 100
        self.layer_2_dim = 100
        self.fc_1_dim = 50
        self.fc_2_dim = 50
        self.dropout = 0.2


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sequence_length = 90
    args = Args()

    model = ModelA_EvolveGCN_w_Dense(args, activation=nn.ReLU, skipfeats=False)
    if device == "cuda:0":
        model.cuda(device=device)

    features = ['adjVolume']
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    epochs = 10
    batch_size = 1
    criterion = nn.CrossEntropyLoss()

    train_data = CompanyStockGraphDataset(features, device=device, start_date='01/01/2010', end_date='31/12/2016')
    # val_data = CompanyStockGraphDataset(features, device=device, start_date='30/09/2016', end_date='31/12/2017')
    # test_data = CompanyStockGraphDataset(features, device=device, start_date='30/09/2017', end_date='31/12/2018')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        print("Epoch: %s" % epoch)
        training_loss = 0
        train_acc = []
        val_loss = 0
        val_acc = []
        for A_train, X_train, mask_train, y_train in tqdm(train_loader):
            model.zero_grad()
            y_pred = model(A_train, X_train, mask_train)
            loss = criterion(y_pred, y_train)
            training_loss +=  loss.item()
            loss.backward()
            optimizer.step()

        for A_val, X_val, mask_val, y_val in tqdm(val_loader):
            y_pred = model(A_val, X_val, mask_val)
            loss = criterion(y_pred, y_val)
            val_loss += loss.item()

        print(f"Training loss: {training_loss}")
        print(f"Validation loss: {val_loss}")