from load_data import Data
import numpy as np
import time
import torch
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import os
import argparse

device = torch.device('cuda:0')

class Experiment:
    def __init__(self, num_iterations, batch_size, learning_rate, decay_rate, ent_vec_dim, rel_vec_dim,
                 k, ni, ranks, input_dropout, hidden_dropout1, hidden_dropout2):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.k = k
        self.ni = ni
        self.ranks = ranks
        self.kwargs = {'input_dropout': input_dropout, 'hidden_dropout1': hidden_dropout1, 'hidden_dropout2': hidden_dropout2}

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets).to(device)
        return np.array(batch), targets

    def evaluate(self, model, data, W):
        hits, ranks, losses = [], [], []
        for _ in [1, 3, 10]:
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0], dtype=torch.long).to(device)
            r_idx = torch.tensor(data_batch[:, 1], dtype=torch.long).to(device)
            e2_idx = torch.tensor(data_batch[:, 2], dtype=torch.long).to(device)
            pred, _ = model.forward(e1_idx, r_idx, W)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = pred[j,e2_idx[j]].item()
                pred[j, filt] = 0.0
                pred[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank+1)
                for id, hits_level in enumerate([1, 3, 10]):
                    if rank+1 <= hits_level:
                        hits[id].append(1.0)
                    else:
                        hits[id].append(0.0)

        return np.mean(1./np.array(ranks)), np.mean(hits[2]), np.mean(hits[1]), np.mean(hits[0])


    def train_and_eval(self):
        print("Training the model...")
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = GETD(d, self.ent_vec_dim, self.rel_vec_dim, self.k, self.ni, self.ranks, device, **self.kwargs)
        model = model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        print("Starting training...")
        best_valid_iter = 0
        best_valid_metric = {'mrr': -1, 'test_mrr': -1, 'test_hit1': -1, 'test_hit3': -1, 'test_hit10': -1}

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)

            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, label = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0], dtype=torch.long).to(device)
                r_idx = torch.tensor(data_batch[:, 1], dtype=torch.long).to(device)

                pred, W = model.forward(e1_idx, r_idx)
                pred = pred.to(device)
                loss = model.loss(pred, label)
                loss.backward()
                opt.step()

                losses.append(loss.item())

            print('\nEpoch %d train, loss=%f' % (it, np.mean(losses, axis=0)))

            if self.decay_rate:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                v_mrr, v_hit10, v_hit3, v_hit1 = self.evaluate(model, d.valid_data, W)
                print(
                    'Epoch %d valid, MRR=%.8f, Hits@10=%f, Hits@3=%f, Hits@1=%f' % (it, v_mrr, v_hit10, v_hit3, v_hit1))
                t_mrr, t_hit10, t_hit3, t_hit1 = self.evaluate(model, d.test_data, W)

                if v_mrr > best_valid_metric['mrr']:
                    best_valid_iter = it
                    print('======== MRR on validation set increases ======== ')
                    best_valid_metric['mrr'] = v_mrr
                    best_valid_metric['test_mrr'] = t_mrr
                    best_valid_metric['test_hit1'] = t_hit1
                    best_valid_metric['test_hit3'] = t_hit3
                    best_valid_metric['test_hit10'] = t_hit10
                else:
                    print(
                        '====Current Epoch:%d, Best Epoch:%d, valid_MRR didn\'t increase for %d Epoch, best test_MRR=%f' % (
                            it, best_valid_iter, it - best_valid_iter, best_valid_metric['test_mrr']))
                print('Epoch %d test, MRR=%.8f, Hits@10=%f, Hits@3=%f, Hits@1=%f' % (
                    it, t_mrr, t_hit10, t_hit3, t_hit1))

            if (it - best_valid_iter) >= 10 or it == self.num_iterations:
                print('++++++++++++ Early Stopping +++++++++++++')
                print('Best epoch %d' % best_valid_iter)
                print('Mean reciprocal rank: {0}'.format(best_valid_metric['test_mrr']))
                print('Hits @10: {0}'.format(best_valid_metric['test_hit10']))
                print('Hits @3: {0}'.format(best_valid_metric['test_hit3']))
                print('Hits @1: {0}'.format(best_valid_metric['test_hit1']))
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k", nargs="?", help="Which dataset to use: WN18 or FB15k.")
    parser.add_argument("--num_iterations", type=int, default=200, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=512, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?", help="Learning rate.")
    parser.add_argument("--dr", type=float, default=0.995, nargs="?", help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?", help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?", help="Relation embedding dimensionality.")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?", help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.5, nargs="?", help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.0, nargs="?", help="Dropout after the second hidden layer.")
    parser.add_argument("--k", type=int, default=3, nargs="?", help="Reshaped tensor order")
    parser.add_argument("--n_i", type=int, default=200, nargs="?", help="Mode-2 dimension of TR latent tensors.")
    parser.add_argument("--TR_ranks", type=int, default=50, nargs="?", help="TR-ranks")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "./data/%s/" % dataset

    torch.backends.cudnn.deterministic = True
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    d = Data(data_dir=data_dir)

    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim,
                            k=args.k, ni=args.n_i, ranks=args.TR_ranks,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, hidden_dropout2=args.hidden_dropout2)
    experiment.train_and_eval()
