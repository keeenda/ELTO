import numpy as np
import torch
import torch.nn as nn
import sys, os
from collections import OrderedDict
sys.path.append(os.path.abspath("./ELTO"))
sys.path.append("..")
# from sklearn.metrics.pairwise import rbf_kernel
# from sklearn.metrics import pairwise
# from gpytorch.kernels import RBFKernel

def _rbf_kernel(X, Y=None, gamma=None):

    if gamma is None:
        gamma = 1.0 / X.shape[1]  # Default gamma value

    if Y is None:
        Y = X

    X = X.float()
    Y = Y.float()

    # Compute pairwise Euclidean distances
    norm_X = torch.sum(X ** 2, dim=1, keepdim=True)
    norm_Y = torch.sum(Y ** 2, dim=1, keepdim=True)
    dist_matrix = norm_X - 2.0 * torch.matmul(X, Y.t()) + norm_Y.t()

    # Compute RBF kernel
    K = torch.exp(-gamma * dist_matrix)

    return K

def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 0.0, torch.exp(x), x + 1.0)
    
class SplitDiagGaussianDecoder(nn.Module):
    def __init__(self, lod: int, out_dim: int):
        super(SplitDiagGaussianDecoder, self).__init__()


        self._hidden_layers_mean, num_last_hidden_mean = self._build_hidden_layers_mean()
        self._hidden_layers_var, num_last_hidden_var = self._build_hidden_layers_var()

        self._out_layer_mean = nn.Linear(in_features=num_last_hidden_mean, out_features=out_dim)
        self._out_layer_var = nn.Linear(in_features=num_last_hidden_var, out_features=out_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _build_hidden_layers_mean(self):

        return nn.ModuleList([
            nn.Linear(in_features=200, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=8)
        ]), 8

    def _build_hidden_layers_var(self):

        return nn.ModuleList([
            nn.Linear(in_features=300, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=8)
        ]), 8

    # def clean_state_dict_keys(self, state_dict):
    #     cleaned_state_dict = OrderedDict()
    #     for key, value in state_dict.items():
    #         if 'hidden_layers_mean' in key:
    #             new_key = key.replace('_module._hidden_layers_mean.', '')
    #             index_offset = 0
    #         elif 'hidden_layers_var' in key:
    #             new_key = key.replace('_module._hidden_layers_var.', '')
    #             index_offset = len(self._hidden_layers_mean)
    #         elif 'out_layer_mean' in key:
    #             new_key = key.replace('_module._out_layer_mean.', str(index_offset + len(self._hidden_layers_mean)) + '.')
    #             index_offset = len(self._hidden_layers_mean) + len(self._hidden_layers_var)
    #         elif 'out_layer_var' in key:
    #             new_key = key.replace('_module._out_layer_var.', str(index_offset + len(self._hidden_layers_mean) + len(self._hidden_layers_var)) + '.')
    #         else:
    #             continue
    #         cleaned_state_dict[new_key] = value
    #
    #     return cleaned_state_dict

    def forward(self, latent_mean, latent_cov):

        h_mean = latent_mean
        for layer in self._hidden_layers_mean:
            h_mean = layer(h_mean)
        mean = self._out_layer_mean(h_mean)

        h_var = latent_cov
        for layer in self._hidden_layers_var:
            h_var = layer(h_var)
        log_var = self._out_layer_var(h_var)
        var = elup1(log_var)

        return mean, var

class MyLayerNorm2d(nn.Module):

    def __init__(self, channels):
        super(MyLayerNorm2d, self).__init__()
        self._scale = torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        self._offset = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        normalized = (x - x.mean(dim=[-3, -2, -1], keepdim=True)) / x.std(dim=[-3, -2, -1], keepdim=True)
        return self._scale * normalized + self._offset

class Deep_Kernel(nn.Module):
    def __init__(self):
        super(Deep_Kernel, self).__init__()

        self.encoder_output_dim = 200
        # idk
        self.observation_dim = 108

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pre_train_enc = torch.load(os.path.abspath("../enc_params.pth"))
        self.pre_train_dec = torch.load(os.path.abspath("../dec_params.pth"))

        self.encoder = self._build_encoder(self.observation_dim, layer_norm=True)
        self.decoder = SplitDiagGaussianDecoder(self.observation_dim, out_dim=8)
        self.mlp = self._build_mlp().to(self.device)
        self.encoder_loaded = False

    def _build_mlp(self):
        return nn.Sequential(
            nn.Linear(self.encoder_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.observation_dim) 
        )

    def _build_encoder(self, input_dim, layer_norm):
        layers = [
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2, stride=2)
        ]
        if layer_norm:
            layers.append(MyLayerNorm2d(channels=12))

        layers.extend([
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, stride=2)
        ])

        if layer_norm:
            layers.append(MyLayerNorm2d(channels=12))

        layers.extend([
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=self.encoder_output_dim),
            nn.ReLU()
        ])

        return nn.Sequential(*layers)

    # def _freeze_encoder(self):
    #     for param in self.encoder.parameters():
    #         param.requires_grad = False
    #
    # def _freeze_all(self):
    #     for param in self.parameters():
    #         param.requires_grad = False
    #     cleaned_state_dict = OrderedDict()
    #     for key, value in state_dict.items():
    #         if 'hidden_layers_mean' in key:
    #             new_key = key.replace('_module._hidden_layers_mean.', '')
    #             index_offset = 0
    #         elif 'hidden_layers_var' in key:
    #             new_key = key.replace('_module._hidden_layers_var.', '')
    #             index_offset = len(self._build_dec_hidden_layers_mean()[0])
    #         elif 'out_layer_mean' in key:
    #             new_key = key.replace('_module._out_layer_mean.', str(index_offset + len(self._build_dec_hidden_layers_mean()[0])) + '.')
    #             index_offset = len(self._build_dec_hidden_layers_mean()[0]) + len(self._build_dec_hidden_layers_var()[0])
    #         elif 'out_layer_var' in key:
    #             new_key = key.replace('_module._out_layer_var.', str(index_offset + len(self._build_dec_hidden_layers_mean()[0]) + len(self._build_dec_hidden_layers_var()[0])) + '.')
    #         else:
    #             continue
    #         cleaned_state_dict[new_key] = value
    #
    #     return cleaned_state_dict

    def _load_model(self):
        self.encoder.load_state_dict(self.pre_train_enc, strict=False)
        # cleaned_state_dict = self.decoder.clean_state_dict_keys(self.pre_train_dec)
        self.decoder.load_state_dict(cleaned_state_dict, strict=False)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder_loaded = True

    def forward(self, obs_1, obs_2=None):

        torch.autograd.set_detect_anomaly(True)

        if self.pre_train_enc and not self.encoder_loaded:
            self._load_model()

        if obs_2 is None:
            encoded_obs = self.encoder(obs_1)
            encoded_obs = encoded_obs.clone()
            rbf_result = _rbf_kernel(encoded_obs)
        else:
            encoded_obs = None
            encoded_obs_1 = self.encoder(obs_1)
            encoded_obs_2 = self.encoder(obs_2)
            encoded_obs_1 = encoded_obs_1.clone()
            encoded_obs_2 = encoded_obs_2.clone()
            rbf_result = _rbf_kernel(encoded_obs_1, encoded_obs_2)

        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = True
        
        return rbf_result, encoded_obs

    def decode(self, latent_mean, latent_cov):
        if not self.encoder_loaded:
            self._load_model()

        return self.decoder(latent_mean, latent_cov)