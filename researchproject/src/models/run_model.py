""" Environment for running models. """
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import numpy as np

from tqdm import tqdm
import pathlib

# from src.models.models import EvolveGCNDenseModel
from src.models.evolvegcn import EvolveGCN
from src.models.models import NodePredictionModel
from src.data.datasets import CompanyStockGraphDataset


class ModelTrainer:

    def __init__(self, gcn, clf, optimizer=optim.SGD, optim_args=None, features=('adjVolume',),
                 criterion=nn.CrossEntropyLoss(), epochs=10):
        if optim_args is None:
            optim_args = {'lr': 0.01, 'momentum': 0.9}

        self.device = device
        self.gcn = gcn
        self.clf = clf
        if self.device == "cuda:0":
            self.gcn.cuda()
            self.clf.cuda()
        self.gcn_optimizer = optim.SGD(self.gcn.parameters(), **optim_args)
        self.clf_optimizer = optim.SGD(self.clf.parameters(), **optim_args)

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

        test_loss, test_acc = self.training_loop(self.test_loader)

    def load_data(self):
        dates = {'train_start': '01/01/2010', 'train_end': '31/12/2016',
                 'val_start': '30/09/2016', 'val_end': '31/12/2017',
                 'test_start': '30/09/2017', 'test_end': '31/12/2018'}

        # dates = {'train_start': '01/01/2015', 'train_end': '31/12/2015',
        #          'val_start': '01/12/2015', 'val_end': '01/02/2016',
        #          'test_start': '30/09/2017', 'test_end': '31/12/2017'}

        train_data = CompanyStockGraphDataset(self.features, device=self.device, start_date=dates['train_start'],
                                              end_date=dates['train_end'], window_size=sequence_length,
                                              predict_periods=predict_periods)
        val_data = CompanyStockGraphDataset(self.features, device=self.device, start_date=dates['val_start'],
                                            end_date=dates['val_end'], window_size=sequence_length,
                                            predict_periods=predict_periods)
        test_data = CompanyStockGraphDataset(self.features, device=self.device, start_date=dates['test_start'],
                                             end_date=dates['test_end'], window_size=sequence_length,

                                             predict_periods=predict_periods)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

    def training_loop(self, loader, training=False):
        running_loss = 0
        mean_loss_hist = []
        acc = []
        pbar = tqdm(loader)

        for i, (*inputs, y_true) in enumerate(pbar):
            self.gcn.zero_grad()
            self.clf.zero_grad()
            node_embs = self.gcn(*inputs)

            y_pred = self.clf(node_embs)

            loss = self.criterion(y_pred, y_true.long())
            # loss = self.criterion(y_pred.view(1, *y_pred.shape).permute(0, 3, 1, 2),
            #                       y_train.view(1, *y_train.shape).long())
            if training:
                loss.backward()
                self.gcn_optimizer.step()
                self.clf_optimizer.step()

            acc.append(get_accuracy(y_true, y_pred))
            running_loss += loss.item()
            mean_loss = running_loss / (i + 1)
            mean_loss_hist.append(mean_loss)

            pbar.set_description(f"Mean loss: {round(mean_loss, 4)}, Mean acc: {round(np.mean(acc))}")

        pbar.close()

        return mean_loss_hist, acc


    def get_node_embs(self, n_embs, n_idx):
        return torch.cat(
            [n_embs[n_set] for n_set in n_idx],
            dim=1
        )

    def plot(self, series_dict, figure_aspect=(8,8)):
        fig, ax = plt.subplots(len(series_dict), figsize=figure_aspect)
        for i, (name, series) in enumerate(series_dict.items()):
            ax[i].plot(series, label=name)
        plt.show()

    # def save_checkpoint(self, state, fn):
    #     torch.save(state, fn)
    #
    # def load_checkpoint(self, fn):
    #     checkpoint = torch.load(fn)
    #     epoch = checkpoint['epoch']
    #     self.gcn.load_state_dict(checkpoint['gcn_dict'])
    #     self.clf.load_state_dict(checkpoint['clf_dict'])
    #     self.gcn_optimizer.load_state_dict(checkpoint['gcn_optimizer'])
    #     self.clf_optimizer.load_state_dict(checkpoint['classifier_optimizer'])
    #     return epoch

    def get_accuracy(self, true, predictions):
        if self.device == "cpu":
            return accuracy_score(true, torch.argmax(predictions, dim=1))
        else:
            return accuracy_score(true.cpu(), torch.argmax(predictions, dim=1).cpu())

class Args:
    def __init__(self):
        self.node_feat_dim = 2
        self.layer_1_dim = 100
        self.layer_2_dim = 100
        self.fc_1_dim = 100
        self.fc_2_dim = 3
        self.dropout = 0.5


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--o', dest="optimizer", default="adam",
                        help="Choice of optimiser (currently 'adam' or 'sgd').")
    parser.add_argument('-epochs', '--e', dest="epochs", default=10, help="# of epochs to run for.")
    parsed, unknown = parser.parse_known_args()

    for arg in unknown:
        if arg.startswith("-opt_"):
            parser.add_argument(arg)
    args = parser.parse_args()

    optim_args = {}
    for arg in args.__dict__:
        if arg.startswith("opt_"):
            # Potential to break if argument contains phrase "opt_"
            try:
                # Will cause issues if args not float...
                optim_args[arg.replace("opt_", "")] = float(args.__dict__[arg])
            except TypeError:
                optim_args[arg.replace("opt_", "")] = args.__dict__[arg]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sequence_length = 30
    predict_periods = 7 #TODO: it's broken @ any value other than 3
    model_args = Args()

    evolve_model = EvolveGCN(model_args, activation=torch.relu, skipfeats=False)
    clf_model = NodePredictionModel(model_args)

    trainer = ModelTrainer(evolve_model, clf_model, optimizer=args.optimizer, optim_args=optim_args, epochs=args.epochs)
    trainer.run()
