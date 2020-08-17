""" Environment for running models. """
import torch
torch.manual_seed(0)

from torch import optim, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sqlite3.dbapi2 import OperationalError
import yaml
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
np.random.seed(0)

from tqdm import tqdm
import pathlib

from src import MODEL_SAVE_DIR, MODEL_ARGS, PG_CREDENTIALS
from src.models.evolvegcn import EvolveGCN
from src.models.lstm import LSTMModel
from src.data.datasets import CompanyStockGraphDataset
from src.data.utils import create_connection_psql

class ModelTrainer:

    # def __init__(self, model, optimizer=optim.SGD, optim_args=None, features=('adjVolume',),
    #              criterion=nn.CrossEntropyLoss(), epochs=10, model_file="gcn_data", clf_file="clf_data",
    #              load_model=None, timeout=30, plot=False):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if args.model == 'egcn':
            self.model = EvolveGCN(args, activation=torch.relu, skipfeats=args.skipfeats)
        elif args.model == 'lstm':
            self.model = LSTMModel(args)
        else:
            raise NotImplementedError("Only 'egcn' and 'lstm' have been implemented so far.")

        self.model_file = MODEL_SAVE_DIR / args.file

        if args.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), **args.optim_args)
        elif args.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **args.optim_args)
        else:
            raise NotImplementedError("Optimizer must be 'sgd' or 'adam'.")

        self.start_epoch = 0
        self.timeout = args.timeout

        if args.load_model:
            # TODO: Make checkpoint loading function
            self.start_epoch = args.load_model.start_epoch
            self.load_checkpoint(self.model, args.load_model.gcn_checkpoint)

        if self.device == "cuda:0":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.model.cuda()

        self.engine = create_connection_psql(PG_CREDENTIALS)
        self.timestamp = time.time()
        self.sequence_length = args.seq_length
        self.predict_periods = args.predict_periods
        self.features = args.features
        self.batch_size = None
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = args.epochs
        self.current_epoch = None
        self.current_iteration = None
        self.returns_threshold = args.returns_threshold
        self.adj = args.adj
        self.adj2 = args.adj2

        if args.dataset == 'small':
            self.dates = {
                'train_start': '01/01/2011', 'train_end': '31/12/2011',
                'val_start': '01/12/2011', 'val_end': '01/02/2012',
                'test_start': '30/09/2013', 'test_end': '31/12/2013'}
        elif args.dataset == 'large':
            self.dates = {
                'train_start': '01/01/2010', 'train_end': '31/12/2016',
                'val_start': '30/09/2016', 'val_end': '31/12/2017',
                'test_start': '30/09/2017', 'test_end': '31/12/2018'}
        else:
            raise NotImplementedError("Dataset must be 'small' or 'large'.")

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def run(self):
        self.load_data(timeout=self.timeout)
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            print("Epoch: %s" % epoch)
            _, train_loss, train_acc = self.training_loop(self.train_loader, training=True)
            _, val_loss, val_acc = self.training_loop(self.val_loader)
            if self.plot:
                self.plot({'Train': train_loss, 'Validation': val_loss})
            print(f"Training loss: {train_loss}")
            print(f"Validation loss: {val_loss}")

        test_predictions, test_loss, test_acc = self.training_loop(self.test_loader)

    def load_data(self, timeout=30):
        self.train_data = CompanyStockGraphDataset(
            self.features, device=self.device, start_date=self.dates['train_start'], end_date=self.dates['train_end'],
            window_size=self.sequence_length, predict_periods=self.predict_periods, timeout=self.timeout,
            returns_threshold=self.returns_threshold, adj=self.adj, adj2=self.adj2
        )
        self.val_data = CompanyStockGraphDataset(
            self.features, device=self.device, start_date=self.dates['val_start'], end_date=self.dates['val_end'],
            window_size=self.sequence_length, predict_periods=self.predict_periods, timeout=self.timeout,
            returns_threshold=self.returns_threshold, adj=self.adj, adj2=self.adj2
        )
        self.test_data = CompanyStockGraphDataset(
            self.features, device=self.device, start_date=self.dates['test_start'], end_date=self.dates['test_end'],
            window_size=self.sequence_length, predict_periods=self.predict_periods, timeout=self.timeout,
            returns_threshold=self.returns_threshold, adj=self.adj, adj2=self.adj2
        )

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def training_loop(self, loader, training=False):
        running_loss = 0
        mean_loss_hist = []
        acc = []
        f1 = []
        prec = []
        rec = []
        with tqdm(loader) as pbar:
            for i, (*inputs, y_true) in enumerate(pbar):
                self.current_iteration = i
                self.model.zero_grad()

                y_pred = self.model(*inputs)

                loss = self.criterion(y_pred, y_true.long())

                if training:
                    loss.backward()
                    self.optimizer.step()

                acc.append(self.get_accuracy(y_true, y_pred))
                # f1.append(self.get_score(f1_score, y_true, y_pred, average='macro'))
                prec.append(self.get_score(precision_score, y_true, y_pred, average='micro'))
                rec.append(self.get_score(recall_score, y_true, y_pred, average='micro'))
                running_loss += loss.item()
                mean_loss = running_loss / (i + 1)
                mean_loss_hist.append(mean_loss)

                pbar.set_description(
                    f"Mean loss: {round(mean_loss, 4)}, Mean acc: {round(np.mean(acc), 4)}, "
                    f"Mean precision: {round(np.mean(prec), 4)}, Mean recall: {round(np.mean(rec), 4)}"
                )
                np_preds = y_pred.argmax(dim=1).numpy()
                print("\n0: ", (np_preds == 0).sum())
                print("1: ", (np_preds == 1).sum())
                print("2: ", (np_preds == 2).sum())
            if training:
                self.save_checkpoint(self.model, self.model_file)

        return y_pred, mean_loss_hist, acc

    # def get_node_embs(self, n_embs, n_idx):
    #     return torch.cat(
    #         [n_embs[n_set] for n_set in n_idx],
    #         dim=1
    #     )

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

    def get_score(self, score, true, predictions, **kwargs):
        if self.device == "cpu":
            return score(true, torch.argmax(predictions, dim=1), **kwargs)
        else:
            return score(true.cpu(), torch.argmax(predictions, dim=1).cpu(), **kwargs)

    def close(self):
        for dataset in (self.train_data, self.test_data, self.val_data):
            if dataset.db == 'sqlite':
                dataset.conn.close()
            else:
                dataset.engine.close()


class Args:
    def __init__(self, arg_file):
        with open(arg_file, 'r') as f:
            arg_dict = yaml.load(f)
        self.__dict__.update(arg_dict)
        # self.node_feat_dim = 2
        # self.layer_1_dim = 200
        # self.layer_2_dim = 200
        # self.fc_1_dim = 100
        # self.fc_2_dim = 3
        # self.dropout = 0.5


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('yaml', default=None, help="Filename for model arguments.")
    parser.add_argument('--load', "-l", dest="load_model", default=None, help="Filename for saved model information.")
    arg = parser.parse_args()

    args = Args((MODEL_ARGS / arg.yaml))
    args.load_model = arg.load_model

    trainer = ModelTrainer(args)
    try:
        trainer.run()
    finally:
        trainer.close()
