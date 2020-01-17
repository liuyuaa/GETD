import numpy as np
import torch
from torch.nn.init import xavier_normal_
import torch.nn.functional as F


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return
    def forward(self, pred1, tar1):
        loss = F.binary_cross_entropy(pred1, tar1)
        return loss

class GETD(torch.nn.Module):
    def __init__(self, d, d_e, d_r, k, ni, ranks, device, **kwargs):
        super(GETD, self).__init__()
        self.E = torch.nn.Embedding(len(d.entities), embedding_dim=d_e, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), embedding_dim=d_r, padding_idx=0)
        self.E.weight.data = (1e-3*torch.randn((len(d.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3*torch.randn((len(d.relations), d_r), dtype=torch.float).to(device))
        self.Zlist = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(np.random.uniform(-1e-1, 1e-1, (ranks, ni, ranks)), dtype=torch.float, requires_grad=True).to(device)) for _ in range(k)])
        self.loss = MyLoss()
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.bne = torch.nn.BatchNorm1d(d_e)
        self.bnr = torch.nn.BatchNorm1d(d_r)
        self.bnw = torch.nn.BatchNorm1d(d_e)
        self.ary = len(d.train_data[0])-1

    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        de = self.E.weight.shape[1]
        dr = self.R.weight.shape[1]
        if W is None:
            k = len(self.Zlist)
            Zlist = [Z for Z in self.Zlist]
            if k == 4:
                W0 = torch.einsum('aib,bjc,ckd,dla->ijkl', Zlist)
            elif k == 5:
                W0 = torch.einsum('aib,bjc,ckd,dle,ema->ijklm', Zlist)
            else:
                print('TR equation is not defined already!')
            if self.ary == 3:
                W = W0.view(dr, de, de, de)
            elif self.ary == 4:
                W = W0.view(dr, de, de, de, de)

        r = self.bnr(self.R(r_idx))
        W_mat = torch.mm(r, W.view(r.size(1), -1))

        if self.ary == 3:
            W_mat = W_mat.view(-1, de, de, de)
            e2, e3 = self.E(e_idx[0]), self.E(e_idx[1])
            e2, e3 = self.bne(e2), self.bne(e3)
            e2, e3 = self.input_dropout(e2), self.input_dropout(e3)
            if miss_ent_domain == 1:  
                W_mat1 = torch.einsum('ijkl,il,ik->ij', W_mat, e3, e2)
            elif miss_ent_domain == 2:
                W_mat1 = torch.einsum('ijkl,il,ij->ik', W_mat, e3, e2)
            elif miss_ent_domain == 3:
                W_mat1 = torch.einsum('ijkl,ij,ik->il', W_mat, e2, e3)
            W_mat1 = self.bnw(W_mat1)
            W_mat1 = self.hidden_dropout(W_mat1)
            x = torch.mm(W_mat1, self.E.weight.transpose(1, 0))

        if self.ary == 4:
            W_mat = W_mat.view(-1, de, de, de, de)
            e2, e3, e4 = self.E(e_idx[0]), self.E(e_idx[1]), self.E(e_idx[2])
            e2, e3, e4 = self.bne(e2), self.bne(e3), self.bne(e4)
            e2, e3, e4 = self.input_dropout(e2), self.input_dropout(e3), self.input_dropout(e4)
            if miss_ent_domain == 1:  
                W_mat1 = torch.einsum('ijklm,il,ik,im->ij', W_mat, e3, e2, e4)
            elif miss_ent_domain == 2:
                W_mat1 = torch.einsum('ijklm,il,ij,im->ik', W_mat, e3, e2, e4)
            elif miss_ent_domain == 3:
                W_mat1 = torch.einsum('ijklm,ij,ik,im->il', W_mat, e2, e3, e4)
            elif miss_ent_domain == 4:
                W_mat1 = torch.einsum('ijklm,ij,ik,il->im', W_mat, e2, e3, e4)
            W_mat1 = self.bnw(W_mat1)
            W_mat1 = self.hidden_dropout(W_mat1)
            x = torch.mm(W_mat1, self.E.weight.transpose(1, 0))

        pred = F.softmax(x, dim=1)

        return pred, W