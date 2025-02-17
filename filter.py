import numpy as np
from torch.linalg import inv
from .deep_kernel import Deep_Kernel, _rbf_kernel
import torch

class ELTO_KF:

    def __init__(self, trained_X, train_obs, window_size, d, device="cuda", is_deep_kernel=False):
        super().__init__()


        self.device = device
        self.is_deep_kernel = is_deep_kernel
        self.n_train = train_obs.shape[0]
        self.h = window_size
        self.N = self.n_train + 1 - 2 * self.h

        self.X = trained_X
        self.y = torch.tensor(train_obs, dtype=torch.float32).to(self.device)


        if len(self.y.shape) == 1:
            self.y = self.y.reshape(-1, 1)


        if len(self.y.shape) == 1:
            self.preimage_states = self.y[self.h:self.h + self.N]
            self.preimage_states = self.preimage_states.reshape(-1, 1)
        else:
            self.preimage_states = self.y[self.h:self.h + self.N, :]

        self.d = d
        self.n_x0 = 200

        self.rbf_kernel = _rbf_kernel
        self.deep_kernel = Deep_Kernel()

        self.kernel_x = self.rbf_kernel

        if self.is_deep_kernel == False:
            self.kernel_y = self.rbf_kernel
        else:
            with torch.no_grad():
                self.kernel_y = self.deep_kernel
                _, self.encoded_y = self.kernel_y(self.preimage_states)

        self.eps_t = np.exp(-6) / (self.N - 1)
        self.eps_o = np.exp(-6) / (self.N - 1)
        self.eps_q = np.exp(-6) / (self.N - 1)

        self.gram_X = None
        self.gram_Y = None
        self.EOO = None
        self.ELTO = None
        self.trans_mat0 = None
        self.transition_err_cov = None
        self.observation_err_cov = None
        self.GO = None
        self.RG = None
        self.YO = None
        self.gram_K0 = None

        # covariance of the error of transition model
        self.transition_err_cov = None
        # covariance of the error of the observation model
        self.observation_err_cov = None

        self.use_observation_err_cov = False

        self.model_learned = False

    def learn_model(self, eps_t=None, eps_o=None, eps_q=None, bandwidth_k=None, bandwidth_g=None):
        # get operator

        # update model parameters
        if eps_t is not None:
            self.eps_t = eps_t
        if eps_o is not None:
            self.eps_o = eps_o
        if eps_q is not None:
            self.eps_q = eps_q

        # compute gram matrices
        self.gram_X = self.kernel_x(self.X.T)
        gram_X1 = self.gram_X[:, :self.N - 1]
        gram_X2 = self.gram_X[:, 1:self.N]
        if self.is_deep_kernel == False:
            self.gram_Y = self.kernel_y(self.preimage_states)
        else:
            self.gram_Y, _ = self.kernel_y(self.preimage_states)

        # initial embeddings
        X0 = torch.randn(self.n_x0, self.d).to(self.device)
        self.gram_K0 = self.kernel_x(self.X[:,:self.N-1].T, X0)

        self.transition_err_cov = 1e-4 * torch.eye(self.N - 1).to(self.device)
        self.observation_err_cov = 1e-5 * torch.eye(self.N).to(self.device)
        self.RG = self.observation_err_cov @ self.gram_Y

        # Embedded Observable Operator
        self.EOO = inv(self.gram_X + self.N * self.eps_o * torch.eye(self.N).T.to(self.device)) @ gram_X2

        # Embedded Latent Transfer Operator
        self.ELTO = inv(self.gram_X[:self.N-1,:self.N-1] + (self.N-1)
                        * self.eps_t * torch.eye(self.N-1).to(self.device)).T @ self.gram_X[:self.N-1,1:self.N]
        # Observation model
        self.GO = self.gram_Y @ self.EOO
        # pre-image step (projection into state space)
        if self.is_deep_kernel == True:
            self.YO = self.encoded_y.T @ self.EOO
        else:
            self.YO = self.preimage_states.T @ self.EOO
        # initial embeddings
        self.trans_mat0 = inv(self.gram_X[:self.N-1, :self.N-1] + (self.N-1)
                              * self.eps_t * torch.eye(self.N-1).to(self.device)).T @ self.gram_K0

        self.model_learned = True

        ELTO = self.ELTO
        EOO = self.EOO

        return ELTO, EOO

    def update_step(self, m, cov, yt, Q=None):
        # update step in paper

        # embed observation y onto RKHS using deep kernel
        if self.is_deep_kernel == False:
            g_Y = self.kernel_y(self.preimage_states, yt)
        else:
            with torch.no_grad():
                _, encoded_preimage = self.kernel_y(self.preimage_states)
                _, encoded_input = self.kernel_y(yt)
                g_Y = _rbf_kernel(encoded_preimage, encoded_input)

        # kernel Kalman gain operator Q
        if Q is None:
            O_cov = self.EOO @ cov
            if self.use_observation_err_cov:
                Q_denominator_T = O_cov @ self.GO.T + self.RG + self.N * self.eps_q * torch.eye(self.N).to(self.device)
            else:
                Q_denominator_T = O_cov @ self.GO.mT + self.N * self.eps_q * torch.eye(self.N).to(self.device)
            
            Q = torch.linalg.solve(Q_denominator_T, O_cov).T
            Q = Q.mT

        # update covariance
        cov = cov - Q @ self.GO @ cov
        # update mean
        m = m + Q @ (g_Y - self.GO @ m)

        return m, cov

    def predict_step(self, m, cov):
        # predict step , aka transition update
        m = self.ELTO @ m

        if cov is not None:
            cov = self.ELTO @ cov @ self.ELTO.T + self.transition_err_cov

            # normalization
            cov = 0.5 * (cov + cov.T)
        return m, cov

    def pre_image(self, m, cov):
        mu = self.YO @ m
        sig = self.YO @ cov @ self.YO.T
        return mu.T, sig

    def initial_transition(self):
        m_t0 = torch.ones((self.n_x0, 1)).to(self.device) / self.n_x0
        S_t0 = torch.eye(self.n_x0).to(self.device)
        m_t = self.trans_mat0 @ m_t0  # 1st transition
        S_t = self.trans_mat0 @ S_t0 @ self.trans_mat0.T + self.transition_err_cov
        return m_t, S_t

    def filter(self, observations):
        assert self.model_learned
        m_t, S_t = self.initial_transition()
        if self.is_deep_kernel == False:
            y_mu_pred = torch.zeros(observations.shape).to(self.device)
            y_sigma_pred = torch.zeros(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
        else:
            y_mu_pred = torch.zeros(observations.shape[0], 200).to(self.device)
            y_sigma_pred = torch.zeros(observations.shape[0], 200, 200).to(self.device)

        for t in range(observations.shape[0]):

            # innovation step / update step
            m_t, S_t = self.update_step(m_t, S_t, observations[t:t+1, :])
            # ignore use_posterior_decoding
            y_mu_pred[t, :], y_sigma_pred[t, :] = self.pre_image(m_t, S_t)

            # predict step / transition update
            m_t, S_t = self.predict_step(m_t, S_t)

        return y_mu_pred, y_sigma_pred