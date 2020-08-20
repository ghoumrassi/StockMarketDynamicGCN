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

from tqdm import tqdm


from src import MODEL_SAVE_DIR, MODEL_ARGS, PG_CREDENTIALS, GEO_DATA
from src.models.evolvegcn import EvolveGCN
from src.models.lstm import LSTMModel
from src.models.dgcn import DGCN
from src.data.datasets import CompanyStockGraphDataset
from src.data.datasets_geo import CompanyGraphDatasetGeo
from src.data.utils import create_connection_psql

class ModelTrainer:

    # def __init__(self, model, optimizer=optim.SGD, optim_args=None, features=('adjVolume',),
    #              criterion=nn.CrossEntropyLoss(), epochs=10, model_file="gcn_data", clf_file="clf_data",
    #              load_model=None, timeout=30, plot=False):
    def __init__(self, args):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.args = args
        if args.model == 'egcn':
            self.model = EvolveGCN(args, activation=torch.relu, skipfeats=args.skipfeats)
            self.geo = False
        elif args.model == 'lstm':
            self.model = LSTMModel(args)
            self.geo = False
        elif args.model == 'dgcn':
            self.model = DGCN(args)
            self.geo = True
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
        self.batch_size = args.batchsize
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = args.epochs
        self.current_epoch = None
        self.current_iteration = None
        self.returns_threshold = args.returns_threshold

        if args.dataset == 'small':
            self.dates = {
                'train_start': '01/01/2010', 'train_end': '30/06/2010',
                'val_start': '01/06/2010', 'val_end': '30/09/2010',
                'test_start': '01/09/2010', 'test_end': '31/12/2010'}
        elif args.dataset == 'medium':
            self.dates = {
                'train_start': '01/01/2011', 'train_end': '31/12/2011',
                'val_start': '01/10/2011', 'val_end': '01/04/2012',
                'test_start': '01/01/2012', 'test_end': '31/12/2012'}
        elif args.dataset == 'large':
            self.dates = {
                'train_start': '01/01/2010', 'train_end': '31/12/2016',
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
        if self.geo:
            self.train_data = CompanyGraphDatasetGeo(
                GEO_DATA, self.features, start_date=self.dates['train_start'], end_date=self.dates['train_end'],
                device=self.device, rthreshold=self.returns_threshold
            )
            self.val_data = CompanyGraphDatasetGeo(
                GEO_DATA, self.features, start_date=self.dates['val_start'], end_date=self.dates['val_end'],
                device=self.device, rthreshold=self.returns_threshold
            )
            self.test_data = CompanyGraphDatasetGeo(
                GEO_DATA, self.features, start_date=self.dates['test_start'], end_date=self.dates['test_end'],
                device=self.device, rthreshold=self.returns_threshold
            )

            self.train_loader = GeoDataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
            self.val_loader = GeoDataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
            self.test_loader = GeoDataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            self.train_data = CompanyStockGraphDataset(
                self.features, device=self.device, start_date=self.dates['train_start'],
                end_date=self.dates['train_end'], window_size=self.sequence_length,
                predict_periods=self.predict_periods, timeout=self.timeout, returns_threshold=self.returns_threshold,
                adj=self.args.adj, adj2=self.args.adj2, k=self.args.k
            )
            self.val_data = CompanyStockGraphDataset(
                self.features, device=self.device, start_date=self.dates['val_start'],
                end_date=self.dates['val_end'], window_size=self.sequence_length,
                predict_periods=self.predict_periods, timeout=self.timeout, returns_threshold=self.returns_threshold,
                adj=self.args.adj, adj2=self.args.adj2, k=self.args.k
            )
            self.test_data = CompanyStockGraphDataset(
                self.features, device=self.device, start_date=self.dates['test_start'],
                end_date=self.dates['test_end'], window_size=self.sequence_length,
                predict_periods=self.predict_periods, timeout=self.timeout, returns_threshold=self.returns_threshold,
                adj=self.args.adj, adj2=self.args.adj2, k=self.args.k
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
            for i, *inputs in enumerate(pbar):
                self.current_iteration = i
                self.model.zero_grad()
                if self.geo:
                    data, slices = inputs[0]
                    batch_size, seq_len = slices['x'].shape
                    seq_len -= 1
                    y_true = data.y.view(batch_size, self.args.seq_length, -1)
                    y_true = y_true[:, -1, :].long()
                    y_pred = self.model((data, slices))
                    loss = self.criterion(y_pred.view(-1, 3), y_true.view(-1))
                else:
                    if self.batch_size:
                        y_preds = []
                        for b in range(self.batch_size):
                            y_pred = self.model(*[inp[b] for inp in inputs[:-1]])
                            y_preds.append(y_pred.reshape(1, *y_pred.shape))

                        loss = self.criterion(torch.cat(y_preds, 0).permute(0, 2, 1), y_true.long())
                    else:
                        y_pred = self.model(*inputs[:-1])
                        loss = self.criterion(y_pred, y_true.long())
                if training:
                    loss.backward()
                    self.optimizer.step()

                # acc.append(self.get_accuracy(y_true, y_pred))
                # # f1.append(self.get_score(f1_score, y_true, y_pred, average='macro'))
                # prec.append(self.get_score(precision_score, y_true, y_pred, average='micro'))
                # rec.append(self.get_score(recall_score, y_true, y_pred, average='micro'))
                running_loss += loss.item()
                mean_loss = running_loss / (i + 1)
                mean_loss_hist.append(mean_loss)

                pbar.set_description(
                    f"Mean loss: {round(mean_loss, 4)}, Mean acc: {round(np.mean(acc), 4)}, "
                    f"Mean precision: {round(np.mean(prec), 4)}, Mean recall: {round(np.mean(rec), 4)}"
                )
                if self.device == "cuda:0":
                    np_preds = y_pred.argmax(dim=2).cpu().numpy()
                else:
                    np_preds = y_pred.argmax(dim=2).numpy()
                print("Preds:")
                print("\n0: ", (np_preds == 0).sum())
                print("1: ", (np_preds == 1).sum())
                print("2: ", (np_preds == 2).sum())
                print("True:")
                print("\n0: ", (y_true.long() == 0).sum().item())
                print("1: ", (y_true.long() == 1).sum().item())
                print("2: ", (y_true.long() == 2).sum().item())
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
