import numpy as np
import torch
from torch.nn.init import xavier_normal_
import torch.nn.functional as F


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return
    def forward(self, pred1, tar1):
        sumloss = F.binary_cross_entropy(pred1, tar1)
        return sumloss


class GETD(torch.nn.Module):
    def __init__(self, d, d_e, d_r, k, ni, ranks, device, **kwargs):
        super(GETD, self).__init__()
        self.E = torch.nn.Embedding(len(d.entities), embedding_dim=d_e, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), embedding_dim=d_r, padding_idx=0)
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3 * torch.randn((len(d.relations), d_r), dtype=torch.float).to(device))
        self.Zlist = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(np.random.uniform(-1e-1, 1e-1, (ranks, ni, ranks)), dtype=torch.float, requires_grad=True).to(device)) for _ in range(k)])
        self.loss = MyLoss()
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.bne = torch.nn.BatchNorm1d(d_e)
        self.bnw = torch.nn.BatchNorm1d(d_e)
        self.bnr = torch.nn.BatchNorm1d(d_r)

    def forward(self, e1_idx, r_idx, W=None):
        d_e = self.E.weight.shape[1]
        d_r = self.R.weight.shape[1]
        if W is None:
            k = len(self.Zlist)
            Zlist = [Z for Z in self.Zlist]
            if k == 3:
                W0 = torch.einsum('aib,bjc,cka->ijk', Zlist)
            elif k == 4:
                W0 = torch.einsum('aib,bjc,ckd,dla->ijkl', Zlist)
            elif k == 5:
                W0 = torch.einsum('aib,bjc,ckd,dle,ema->ijklm', Zlist)
            else:
                print('TR equation is not defined already!')
            W = W0.view(d_e, d_r, d_e)

        r = self.R(r_idx)
        r = self.input_dropout(self.bnr(r))
        r = r.view(-1, 1, d_r)
        e1 = self.bne(self.E(e1_idx))

        W_mat = torch.mm(e1, W.view(e1.size(1), -1))
        W_mat = W_mat.view(-1, d_r, d_e)
        W_mat = self.hidden_dropout1(W_mat)

        W_mat1 = (torch.bmm(r, W_mat)).view(-1, d_e)
        W_mat1 = self.bnw(W_mat1)
        W_mat1 = self.hidden_dropout2(W_mat1)
        x = torch.mm(W_mat1, self.E.weight.transpose(1, 0))
        pred = F.softmax(x, dim=1)

        return pred, W