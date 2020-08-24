import torch
from sqlalchemy import text
import datetime as dt
from src import QUERIES


def pad_with_last_col(matrix, cols):
    out = [matrix]
    pad = [matrix[:, [-1]]] * (cols - matrix.size(1))
    out.extend(pad)
    return torch.cat(out, dim=1)


def pad_with_last_val(vect, k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                     dtype=torch.long,
                     device=device) * vect[-1]
    vect = torch.cat([vect, pad])
    return vect


class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)


def normalize(x):
    means = x.mean(dim=0, keepdim=True)
    stds = x.std(dim=0, keepdim=True)
    return (x - means) / stds


def get_ce_weights(engine, start, end, threshold):
    start = dt.datetime.strptime(start, '%d/%m/%Y').timestamp()
    end = dt.datetime.strptime(end, '%d/%m/%Y').timestamp()
    with open((QUERIES / 'psql' / 'class_weights.q'), 'r') as f:
        weights_query = f.read()
    resultset = engine.execute(text(weights_query), startdate=int(start), enddate=int(end), threshold=threshold)
    results = resultset.fetchall()
    out = torch.tensor(results)[:, 1]
    return out.true_divide(out.sum())

if __name__ == "__main__":
    from src.data.utils import create_connection_psql
    from src import PG_CREDENTIALS
    engine = create_connection_psql(PG_CREDENTIALS)
    get_ce_weights(engine, "01/01/2009", "01/01/2019", 0.01)
