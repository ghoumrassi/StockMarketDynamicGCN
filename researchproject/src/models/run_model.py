""" Environment for running models. """
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from tqdm import tqdm
import pathlib

from src import MODEL_SAVE_DIR, logger
from src.models.evolvegcn import EvolveGCN
from src.models.models import NodePredictionModel
from src.data.datasets import CompanyStockGraphDataset


class ModelTrainer:

    def __init__(self, gcn, clf, optimizer=optim.SGD, optim_args=None, features=('adjVolume',),
                 criterion=nn.CrossEntropyLoss(), epochs=10, gcn_file="gcn_data", clf_file="clf_data",
                 load_model=None):
        if optim_args is None:
            optim_args = {'lr': 0.01, 'momentum': 0.9}

        self.device = device
        self.gcn = gcn
        self.clf = clf
        self.gcn_file = MODEL_SAVE_DIR / gcn_file
        self.clf_file = MODEL_SAVE_DIR / clf_file
        self.model_files = [self.gcn_file, self.clf_file]
        self.start_epoch = 0

        if load_model:
            # TODO: Make checkpoint loading funtion
            self.start_epoch = load_model.start_epoch
            self.load_checkpoint(self.gcn, load_model.gcn_checkpoint)
            self.load_checkpoint(self.clf, load_model.clf_checkpoint)
            pass

        if self.device == "cuda:0":
            self.gcn.cuda()
            self.clf.cuda()
        self.gcn_optimizer = optim.SGD(self.gcn.parameters(), **optim_args)
        self.clf_optimizer = optim.SGD(self.clf.parameters(), **optim_args)

        self.features = features
        self.batch_size = None
        self.criterion = criterion
        self.epochs = epochs
        self.current_epoch = None
        self.current_iteration = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def run(self):
        self.load_data()
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
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

        dates = {'train_start': '01/01/2015', 'train_end': '31/12/2015',
                 'val_start': '01/12/2015', 'val_end': '01/02/2016',
                 'test_start': '30/09/2017', 'test_end': '31/12/2017'}

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
        f1 = []
        pbar = tqdm(loader)

        try:
            for i, (*inputs, y_true) in enumerate(pbar):
                self.current_iteration = i
                self.gcn.zero_grad()
                self.clf.zero_grad()
                node_embs = self.gcn(*inputs)

                y_pred = self.clf(node_embs)

                loss = self.criterion(y_pred, y_true.long())

                if training:
                    loss.backward()
                    self.gcn_optimizer.step()
                    self.clf_optimizer.step()

                acc.append(self.get_accuracy(y_true, y_pred))
                f1.append(self.get_f1(y_true, y_pred))
                running_loss += loss.item()
                mean_loss = running_loss / (i + 1)
                mean_loss_hist.append(mean_loss)

                pbar.set_description(
                    f"Mean loss: {round(mean_loss, 4)}, Mean acc: {round(np.mean(acc), 4)}, "
                    f"Mean F1: {round(np.mean(f1), 4)}"
                )

            if training:
                self.save_checkpoint(self.gcn, self.gcn_file)
                self.save_checkpoint(self.clf, self.clf_file)

            pbar.close()
        except Exception as e:
            logger.log_model_error(e, self.model_files, self.current_epoch, self.current_iteration)
            pbar.close()
            raise Exception("Error occured: check 'modelerror' table.")

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

    def save_checkpoint(self, state, fn):
        torch.save(state, fn)

    def load_checkpoint(self, model, fn):
        checkpoint = torch.load(fn)
        model = model.load_state_dict(checkpoint)
        return model

    def get_accuracy(self, true, predictions):
        if self.device == "cpu":
            return accuracy_score(true, torch.argmax(predictions, dim=1))
        else:
            return accuracy_score(true.cpu(), torch.argmax(predictions, dim=1).cpu())

    def get_f1(self, true, predictions):
        if self.device == "cpu":
            return f1_score(true, torch.argmax(predictions, dim=1), average="macro")
        else:
            return f1_score(true.cpu(), torch.argmax(predictions, dim=1).cpu(), average="macro")


class Args:
    def __init__(self):
        self.node_feat_dim = 2
        self.layer_1_dim = 512
        self.layer_2_dim = 512
        self.fc_1_dim = 200
        self.fc_2_dim = 3
        self.dropout = 0.5


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--o', dest="optimizer", default="adam",
                        help="Choice of optimiser (currently 'adam' or 'sgd').")
    parser.add_argument('-epochs', '--e', dest="epochs", default=10, type=int, help="# of epochs to run for.")
    parser.add_argument('-load', '--l', dest="load_model", default=None, help="Filename for loaded model.")
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
    sequence_length = 90
    predict_periods = 3 #TODO: it's broken @ any value other than 3
    model_args = Args()

    evolve_model = EvolveGCN(model_args, activation=torch.relu, skipfeats=False)
    clf_model = NodePredictionModel(model_args)

    if args.optimizer == "sgd":
        optimizer = optim.SGD
    elif args.optimizer == "adam":
        optimizer = optim.Adam
    else:
        raise NotImplementedError("Optimizer must be 'sgd' or 'adam'.")

    trainer = ModelTrainer(evolve_model, clf_model, optimizer=optimizer, optim_args=optim_args, epochs=args.epochs,
                           load_model=args.load_model)
    trainer.run()
