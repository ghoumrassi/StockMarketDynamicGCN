""" Environment for running models. """
import torch
torch.manual_seed(0)

from torch import optim, nn
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeoDataLoader
import matplotlib.pyplot as plt
from sqlite3.dbapi2 import OperationalError
import yaml
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
np.random.seed(0)
from sqlalchemy import text
from tqdm import tqdm


from src import MODEL_SAVE_DIR, MODEL_ARGS, PG_CREDENTIALS, GEO_DATA
from src.models.evolvegcn import EvolveGCN
from src.models.evolve_geo import Evolve
from src.models.lstm import LSTMModel
from src.models.dgcn import *
from src.data.datasets import CompanyStockGraphDataset
from src.data.datasets_geo import CompanyGraphDatasetGeo
from src.data.dataset_elliptic_temporal import EllipticTemporalDataset
from src.data.utils import create_connection_psql
from src.models.utils import get_ce_weights


class ModelTrainer:
    def __init__(self, args, log=True, test=False):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.args = args
        if args.model == 'evolve':
            self.model = Evolve(args)
        elif args.model == 'temporal':
            self.model = LSTMModel(args, device=self.device)
        elif args.model == 'dgcn':
            self.model = DGCN(args, device=self.device)
        elif args.model == 'dgcn_agg':
            self.model = DGCNAgg(args, device=self.device)
        elif args.model == 'dgcn2':
            self.model = DGCN2(args, device=self.device)
        else:
            raise NotImplementedError("The chosen model doesn't exist.")

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
            self.start_epoch = int(input("Start epoch: "))
            self.load_checkpoint(self.model, (MODEL_SAVE_DIR / args.load_model))

        if self.device == "cuda:0":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.model.cuda()

        self.log = log
        if self.log:
            self.model_name = args.name
        self.engine = create_connection_psql(PG_CREDENTIALS)
        self.timestamp = time.time()
        self.sequence_length = args.seq_length
        self.predict_periods = args.predict_periods
        self.features = args.features
        self.batch_size = args.batchsize
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = args.epochs
        self.current_epoch = None
        self.current_iteration = None
        self.returns_threshold = args.returns_threshold
        self.phase = None
        self.test = test
        if 'conv_first' in args.__dict__:
            self.conv_first = args.conv_first
        else:
            self.conv_first = True

        if args.size == 'small':
            self.dates = {
                # 'train_start': '01/01/2010', 'train_end': '30/06/2010',
                'train_start': '01/06/2010', 'train_end': '01/06/2011',
                'val_start': '01/06/2010', 'val_end': '01/06/2011',
                'test_start': '01/06/2010', 'test_end': '01/06/2011'}
        elif args.size == 'medium':
            self.dates = {
                'train_start': '01/01/2011', 'train_end': '31/12/2011',
                'val_start': '01/10/2011', 'val_end': '01/04/2012',
                'test_start': '01/01/2012', 'test_end': '31/12/2012'}
        elif args.size == 'large':
            self.dates = {
                'train_start': '01/06/2010', 'train_end': '31/12/2016',
                'val_start': '30/09/2016', 'val_end': '31/12/2017',
                'test_start': '30/09/2017', 'test_end': '31/12/2018'}
        else:
            raise NotImplementedError("Dataset must be 'small', 'medium' or 'large'.")

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def run(self):
        self.load_data(timeout=self.timeout)
        self.criterion.weight = get_ce_weights(self.engine, self.dates['train_start'], self.dates['train_end'],
                                               self.args.returns_threshold).to(self.device)
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            print("Epoch: %s" % epoch)
            self.phase = 'training'
            _, train_loss, train_acc = self.training_loop(self.train_loader, training=True)
            self.phase = 'validation'
            _, val_loss, val_acc = self.training_loop(self.val_loader)
            if self.args.plot:
                self.args.plot({'Train': train_loss, 'Validation': val_loss})
            print(f"Training loss: {train_loss}")
            print(f"Validation loss: {val_loss}")
        self.phase = 'testing'
        test_predictions, test_loss, test_acc = self.training_loop(self.test_loader)

    def load_data(self, timeout=30):
        if self.args.dataset == 'main':
            self.train_data = CompanyGraphDatasetGeo(
                GEO_DATA, self.features, start_date=self.dates['train_start'], end_date=self.dates['train_end'],
                device=self.device, rthreshold=self.returns_threshold, test=self.test, periods=self.predict_periods,
                edgetypes=self.args.edgetypes, conv_first=self.conv_first
            )
            self.val_data = CompanyGraphDatasetGeo(
                GEO_DATA, self.features, start_date=self.dates['val_start'], end_date=self.dates['val_end'],
                device=self.device, rthreshold=self.returns_threshold, test=self.test, periods=self.predict_periods,
                edgetypes=self.args.edgetypes, conv_first=self.conv_first
            )
            self.test_data = CompanyGraphDatasetGeo(
                GEO_DATA, self.features, start_date=self.dates['test_start'], end_date=self.dates['test_end'],
                device=self.device, rthreshold=self.returns_threshold, test=self.test, periods=self.predict_periods,
                edgetypes=self.args.edgetypes, conv_first=self.conv_first
            )
            self.train_loader = GeoDataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
            self.val_loader = GeoDataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
            self.test_loader = GeoDataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        elif self.args.dataset == 'elliptic':
            self.train_data = EllipticTemporalDataset(device=self.device)
            self.val_data = EllipticTemporalDataset(device=self.device)
            self.test_data = EllipticTemporalDataset(device=self.device)
            self.train_loader = GeoDataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
            self.val_loader = GeoDataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
            self.test_loader = GeoDataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            raise NotImplementedError("The dataset chosen has not been implemented.")

    def training_loop(self, loader, training=False):
        running_loss = 0
        mean_loss_hist = []
        acc = []
        f1 = []
        prec = []
        rec = []
        profit = []
        with tqdm(loader) as pbar:
            for i, data in enumerate(pbar):
                self.current_iteration = i
                self.model.zero_grad()

                if self.conv_first:
                    y_true = data.y.view(self.batch_size, self.sequence_length, -1)
                    y_true = y_true[:, -1, :].long()
                    r = data.r.view(self.batch_size, self.sequence_length, -1)
                else:
                    y_true = data.y.view(self.batch_size, -1).long()
                    r = data.r.view(self.batch_size, -1)
                y_pred = self.model(data)
                loss = self.criterion(y_pred.view(-1, 3), y_true.view(-1))

                if training:
                    loss.backward()
                    self.optimizer.step()

                acc.append(self.get_accuracy(y_true, y_pred))
                f1.append(self.get_score(f1_score, y_true, y_pred, average='weighted'))
                prec.append(self.get_score(precision_score, y_true, y_pred, average='weighted'))
                rec.append(self.get_score(recall_score, y_true, y_pred, average='weighted'))
                profit.append(self.get_profit(r, y_pred))
                running_loss += loss.item()
                mean_loss = running_loss / (i + 1)
                mean_loss_hist.append(mean_loss)

                pbar.set_description(
                    f"Mean loss: {round(mean_loss, 4)}, Mean acc: {round(np.mean(acc), 4)}, "
                    f"Mean precision: {round(np.mean(prec), 4)}, Mean recall: {round(np.mean(rec), 4)}, "
                    f"Mean profit: {round(np.mean(profit), 4)}"
                )
                if self.device == "cuda:0":
                    np_preds = y_pred.argmax(dim=1).cpu().numpy()
                else:
                    np_preds = y_pred.argmax(dim=1).numpy()

                print("\nPreds:",
                      "\n0: ", (np_preds == 0).sum(),
                      ", 1: ", (np_preds == 1).sum(),
                      ", 2: ", (np_preds == 2).sum())
                print("True:",
                      "\n0: ", (y_true.long() == 0).sum().item(),
                      "1: ", (y_true.long() == 1).sum().item(),
                      "2: ", (y_true.long() == 2).sum().item())

                if self.log:
                    log_data = {
                        'name': self.model_name, 'loss': loss.item(), 'accuracy': acc[-1], 'f1': f1[-1],
                        'precision': prec[-1], 'recall': rec[-1], 'profit': profit[-1], 'epoch': self.current_epoch,
                        'iteration': i, 'phase': self.phase
                    }
                    self.log_metrics(log_data)

            if training:
                self.save_checkpoint(self.model, self.model_file)

        return y_pred, mean_loss_hist, acc

    def plot(self, series_dict, figure_aspect=(8,8)):
        fig, ax = plt.subplots(len(series_dict), figsize=figure_aspect)
        for i, (name, series) in enumerate(series_dict.items()):
            ax[i].plot(series, label=name)
        plt.show()

    def save_checkpoint(self, state, fn):
        torch.save(state, fn)

    def load_checkpoint(self, model, fn):
        model = torch.load(fn)
        # model = model.load_state_dict(checkpoint)
        return model

    def log_metrics(self, data):
        q = """INSERT INTO model_logs (name, loss, accuracy, f1, precision, recall, profit, epoch, iteration, phase)
        values (:name, :loss, :accuracy, :f1, :precision, :recall, :profit, :epoch, :iteration, :phase)"""
        self.engine.execute(text(q), **data)

    def get_accuracy(self, true, predictions):
        true = true.reshape(-1)
        predictions = predictions.reshape(-1, 3)
        if self.device == "cpu":
            return accuracy_score(true, torch.argmax(predictions, dim=1))
        else:
            return accuracy_score(true.cpu(), torch.argmax(predictions, dim=1).cpu())

    def get_score(self, score, true, predictions, **kwargs):
        true = true.reshape(-1)
        predictions = predictions.reshape(-1, 3)
        if self.device == "cpu":
            return score(true, torch.argmax(predictions, dim=1), **kwargs)
        else:
            return score(true.cpu(), torch.argmax(predictions, dim=1).cpu(), **kwargs)

    def get_profit(self, returns, predictions):
        if self.conv_first:
            returns = returns[:, -1, :].reshape(-1)
        else:
            returns = returns.reshape(-1)
        predictions = torch.argmax(predictions.reshape(-1, 3), dim=1)
        return returns[predictions == 2].mean().item()

    def close(self):
        for dataset in (self.train_data, self.test_data, self.val_data):
            dataset.engine.close()


class Args:
    def __init__(self, arg_file):
        with open(arg_file, 'r') as f:
            arg_dict = yaml.load(f)
        self.__dict__.update(arg_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('yaml', default=None, help="Filename for model arguments.")
    parser.add_argument('--name', dest="name", default='model', help="Name of model in model logs.")
    parser.add_argument('--load', "-l", dest="load_model", default=None, help="Filename for saved model information.")
    parser.add_argument('--no-log', dest="log", action='store_false', help="Disables logging of training metrics")
    parser.add_argument('--test', dest="test", action='store_true', help="Testing mode: dataset will not be processed")

    arg = parser.parse_args()

    args = Args((MODEL_ARGS / arg.yaml))
    args.load_model = arg.load_model
    args.name = arg.name

    trainer = ModelTrainer(args, arg.log, arg.test)

    trainer.run()