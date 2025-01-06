
import numpy as np
import torch
import torch.nn as nn
from torch.linalg import svd, inv
from torchvision import transforms
import torch.nn.functional as F
from sklearn import rbf_kernel
from .deep_kernel import Deep_Kernel, _rbf_kernel

class ELTO_Kernel(nn.Module):
    def __init__(self, observations, epochs, batch_size, data_window_size, spectral_window_size, device="cuda"):

        # window_size : h in the paper, length of elements to analyse at the same time
        # d : as described in the paper, top d canonical correlations, the first few elements on the diag

        super(ELTO_Kernel, self).__init__()
        self.y = torch.tensor(observations, dtype=torch.float32).to(device)
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(1)

        self.batches_per_epoch = int(observations.shape[0] / batch_size)
        # n_train : T
        self.n_train = observations.shape[0]  # seq_length

        # h : window size
        self.h = data_window_size
        # N = T + 2 - 2h
        self.N = self.n_train + 1 - 2 * self.h
        self.d = spectral_window_size
        self.device = device

        self.grammat_all = None
        self.grammat_ff = torch.zeros(self.h, self.h, self.n_train, self.n_train).to(device)
        self.grammat_pp = torch.zeros(self.h, self.h, self.n_train, self.n_train).to(device)
        self.grammat_fp = torch.zeros(self.h, self.h, self.n_train, self.n_train).to(device)

        self.epochs = epochs
        self.batch_size = batch_size

        # initialize parameter w and deep kernel
        self.w = nn.Parameter((torch.ones(self.n_train,) / self.n_train).to(device), requires_grad= True)
        self.deep_kernel = Deep_Kernel()

    def epsvd(self, Cff, Cpp, Cfp):
        # eigen-probability svd : inspired by ALg A @ section 8.7 of the book
        # Subspace methods for system identification, Katayama, 2005
        # inputs : empirical covariance matrices Cff, Cpp, Cfp described in paper
        # capital sigma ff/fp/pp in Alg A
        # outputs : SVD result of OC = L^(-1) * Cfp * M^(-T)

        # step 1 compute the square root of Cff and Cpp
        Uff, Sff, Vff = svd(Cff)
        Upp, Spp, Vpp = svd(Cpp)
        Sf = torch.sqrt(torch.diag(Sff))
        Sp = torch.sqrt(torch.diag(Spp))

        # obtain L, M, and their inverse
        # where Cff = L * L^T ,  Cpp = M * M^T
        L = Uff @ Sf @ Vff
        M = Upp @ Sp @ Vpp
        Linv = Vff @ inv(Sf) @ Uff.T
        Minv = Vpp @ inv(Sp) @ Upp.T

        # step 2 calculate SVD of L^(-1)*Cfp*M^(-T) (denoted OC)
        OC = Linv @ Cfp @ Minv.T
        UU, SS, VV = svd(OC)

        # we return M^(-1) since it will be used in the next algorithm
        return UU, SS, VV, L, M, Minv

    def cca_train_loss(self):

        # grammat are the embedding matrices in the paper
        # d means the first d-th canonical correlations
        # which determines the length of the period to train
        Nb = self.batch_size
        Cff = self.grammat_ff @ self.w @ self.w / (Nb)
        Cpp = self.grammat_pp @ self.w @ self.w / (Nb)
        Cfp = self.grammat_fp @ self.w @ self.w / (Nb)

        UU, SS, VV, L, M, Minv= self.epsvd(Cff, Cpp, Cfp)

        # choose sum of the first a few correlation as loss func
        # to maximize in order to get the most spectral correlation w.r.t. time
        return -sum(SS[:self.d]) ,SS

    def cca_loss_entropy(self):
        pass

    def data_ids_stream(self):
        rng = np.random.RandomState(0) 
        while True:
             perm = rng.permutation(self.N)
             for i in range(self.batches_per_epoch):
                batch_idx = perm[i * self.batch_size: (i + 1) * self.batch_size]
                yield batch_idx

    def update(self, batch_ids):
        for i in range(self.h):
            p_i_ids = (self.h - 1 - i) + batch_ids
            f_i_ids = (self.h + i) + batch_ids
            for j in range(self.h):
                p_j_ids = (self.h - 1 - j) + batch_ids
                f_j_ids = (self.h + j) + batch_ids
                self.grammat_pp[i, j, :, :] = torch.mm(self.grammat_all[:, p_i_ids],self.grammat_all[p_j_ids, :])
                self.grammat_fp[i, j, :, :] = torch.mm(self.grammat_all[:, f_i_ids],self.grammat_all[p_j_ids, :])
                self.grammat_ff[i, j, :, :] = torch.mm(self.grammat_all[:, f_i_ids],self.grammat_all[f_j_ids, :])
        return


    def get_trained_w(self, is_deep_kernel=False):
        # if  is_deep_kernel == False:
        # rbf kernel
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.grammat_all = rbf_kernel(self.y)

        for itr in range(self.epochs):
            optimizer.zero_grad()
            batch_ids = self.data_ids_stream()
            self.update(next(batch_ids))
            loss, SS = self.cca_train_loss()
            loss.backward()
            # print(f'"SVD result", {SS}')
            nn.utils.clip_grad_norm_(ELTO_Kernel.parameters(self), max_norm=1)
            optimizer.step()
            if itr == 0:
                print(f'{"itr       CCA loss"}')
            if (itr + 1) % 20 == 0:
                print(f'{itr + 1}, {-loss}')
        print('trained_w get! \n')


    def get_trained_w(self, is_deep_kernel=False):
        
        if  is_deep_kernel == False:

            # rbf kernel
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            self.grammat_all = _rbf_kernel(self.y)

            for itr in range(self.epochs):
                optimizer.zero_grad()
                batch_ids = self.data_ids_stream()
                self.update(next(batch_ids))
                loss, SS = self.cca_train_loss()
                loss.backward()
                # print(f'"SVD result", {SS}')
                nn.utils.clip_grad_norm_(ELTO_Kernel.parameters(self), max_norm=1)
                optimizer.step()
                if itr == 0:
                    print(f'{"itr       CCA loss"}')
                if (itr + 1) % 20 == 0:
                    print(f'{itr + 1}, {-loss}')
            print('trained_w get! \n')

        else:
            # use deep kernel
            optimizer = torch.optim.Adam(self.deep_kernel.parameters(), lr=1e-3)
            optimizer.add_param_group({'params': self.w, 'lr': 1e-3})
            y = self.y

            self.deep_kernel.train()
            self.deep_kernel._freeze_encoder()

            for itr in range(self.epochs):
                self.grammat_all = self.deep_kernel(y)[0].detach()
                optimizer.zero_grad()
                batch_ids = self.data_ids_stream()
                self.update(next(batch_ids))
                loss, SS = self.cca_train_loss()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(ELTO_Kernel.parameters(self), max_norm=1)
                optimizer.step()
                if itr == 0:
                    print(f'{"itr        CCA loss"}')
                if (itr + 1) % 20 == 0:
                    print(f'{itr + 1}   {-loss}')
            print('trained_w get! \n')

    def get_trained_x(self):
        h = self.h
        n_train = self.n_train
        N = self.N
        d = self.d

        gram_mat_for_pp = torch.zeros((h, h, n_train, n_train,)).to(self.device)
        gram_mat_for_fp = torch.zeros((h, h, n_train, n_train,)).to(self.device)
        gram_mat_for_ff = torch.zeros((h, h, n_train, n_train,)).to(self.device)
        for i in range(h):
            p_i1, p_i2 = (h - 1 - i), (h - 1 - i) + N
            f_i1, f_i2 = h + i, h + N + i
            for j in range(h):
                p_j1, p_j2 = (h - 1 - j), (h - 1 - j) + N
                f_j1, f_j2 = h + j, h + N + j
                gram_mat_for_pp[i, j, :, :] = self.grammat_all[:, p_i1:p_i2] @ self.grammat_all[p_j1:p_j2, :]
                gram_mat_for_fp[i, j, :, :] = self.grammat_all[:, f_i1:f_i2] @ self.grammat_all[p_j1:p_j2, :]
                gram_mat_for_ff[i, j, :, :] = self.grammat_all[:, f_i1:f_i2] @ self.grammat_all[f_j1:f_j2, :]
        Rff = gram_mat_for_ff @ self.w @ self.w / (N * N)
        Rpp = gram_mat_for_fp @ self.w @ self.w / (N * N)
        Rfp = gram_mat_for_pp @ self.w @ self.w / (N * N)

        _, SS, VV, _, _, Minv = self.epsvd(Rff, Rfp, Rpp)
        print(f'\n SS[:d]={SS[:d].data}')
        print(f'corr = {torch.sum(SS[:d]).data}, trained_x get! \n')

        S1 = torch.diag(torch.sqrt(SS[:d]))
        V1 = VV[:, :d]
        A = S1 @ V1.T @ Minv # B

        # calculate the sequence of x
        X = torch.zeros((h, N,)).to(self.device)
        for t in range(N):
            i1 = h + t
            i2 = t
            X[:, t] = A @ torch.flip(self.grammat_all[i2:i1, :], dims=[0]) @ self.w
        ss = preprocessing.StandardScaler()
        X = torch.tensor(ss.fit_transform(X.detach().cpu().numpy().T)).T.to(self.device)
        return X

    def forward(self, is_deep_kernel):
        self.get_trained_w(is_deep_kernel)
        trained_x = self.get_trained_x()
        return trained_x