""" Environment for running models. """
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.models.models import EvolveGCNDenseModel
from src.data.datasets import CompanyStockGraphDataset


class ModelTrainer:
    def __init__(self, model, optimizer=optim.SGD, optim_args=None, features=('adjVolume',),
                 criterion=nn.CrossEntropyLoss(), epochs=10):
        if optim_args is None:
            optim_args = {'lr': 0.01}

        self.model = model
        if device == "cuda:0":
            self.model.cuda()
        self.optimizer = optimizer(self.model.params, **optim_args)
        self.features = features
        self.batch_size = None
        self.criterion = criterion
        self.epochs = epochs

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def run(self):
        self.load_data()
        for epoch in range(self.epochs):
            print("Epoch: %s" % epoch)
            train_loss, train_acc = self.training_loop(self.train_loader, training=True)
            val_loss, val_acc = self.training_loop(self.val_loader)
            self.plot({'Train': train_loss, 'Validation': val_loss})
            print(f"Training loss: {train_loss}")
            print(f"Validation loss: {val_loss}")

    def load_data(self):
        train_data = CompanyStockGraphDataset(self.features, device=device, start_date='01/01/2010',
                                              end_date='31/12/2016', window_size=sequence_length,
                                              predict_periods=predict_periods)
        val_data = CompanyStockGraphDataset(self.features, device=device, start_date='30/09/2016',
                                            end_date='31/12/2017', window_size=sequence_length,
                                            predict_periods=predict_periods)
        test_data = CompanyStockGraphDataset(self.features, device=device, start_date='30/09/2017',
                                             end_date='31/12/2018', window_size=sequence_length,
                                             predict_periods=predict_periods)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

    def training_loop(self, loader, training=False):
        running_loss = 0
        mean_loss_hist = []
        acc = []
        pbar = tqdm(loader)
        for i, (*inputs, y_train) in enumerate(pbar):
            self.model.zero_grad()
            y_pred = self.model(*inputs)
            loss = self.criterion(y_pred.view(1, *y_pred.shape).permute(0, 3, 1, 2),
                                  y_train.view(1, *y_train.shape).long())
            running_loss += loss.item()
            mean_loss = running_loss / (i + 1)
            mean_loss_hist.append(mean_loss)
            if training:
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f"Mean loss: {round(mean_loss, 4)}")
            if i == 10:
                break
        pbar.close()

        return mean_loss_hist, acc

    def plot(self, series_dict, figure_aspect=(8,8)):
        fig, ax = plt.subplots(len(series_dict), figsize=figure_aspect)
        for i, (name, series) in enumerate(series_dict.items()):
            ax[i].plot(series, label=name)
        plt.show()

class Args:
    def __init__(self):
        self.node_feat_dim = 2
        self.layer_1_dim = 100
        self.layer_2_dim = 100
        self.fc_1_dim = 50
        self.fc_2_dim = 3
        self.dropout = 0.2


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sequence_length = 30
    predict_periods = 3
    args = Args()

    evolve_model = EvolveGCNDenseModel(args, activation=torch.relu, skipfeats=False, predict_periods=3)

    trainer = ModelTrainer(evolve_model)
    trainer.run()
