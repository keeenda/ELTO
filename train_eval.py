import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision import models
from .spectral_learning import ELTO_Kernel
from .filter import ELTO_KF
from .modedecom import ELTO_MD
from deep_kernel import Deep_Kernel

class Eval_pendulum():
    def __init__(self):
        super().__init__()

        self.__test_observations = None
        self.__test_groundtruth = None

        self.train_model_class = ELTO_Kernel
        self.eval_model_class = ELTO_KF
        self.train_model = None
        self.eval_model = None

        self.device = "cuda"
        self._is_setup = False

        self.trained_x = None

    def setup_training(self, train_obs, test_observations, test_groundtruth, epochs, batch_size, window_size, d):
    
        self.train_model = self.train_model_class(train_obs, epochs, batch_size, window_size, d)
        self.trained_x = self.train_model.forward(is_deep_kernel=False)
        self.eval_model = self.eval_model_class(self.trained_x, train_obs, window_size, d)

        self.__test_observations = torch.tensor(test_observations, dtype=torch.float32).to(self.device)
        self.__test_groundtruth = torch.tensor(test_groundtruth, dtype=torch.float32) .to(self.device)

        self._is_setup = True

    def evaluate(self, eval_function=mean_squared_error, **kwargs):
        assert self.trained_x is not None
        assert self._is_setup

        # compute ELTO and EOO
        self.eval_model.learn_model(**kwargs)
        # evaluation
        mu, sigma = self.eval_model.filter(self.__test_observations)
        eval_loss = eval_function(self.__test_groundtruth[1:, :], mu[1:, :], sigma)
        return eval_loss.item()
     
def mean_squared_error(groundtruth, mu, _):
    return torch.mean((groundtruth - mu) ** 2)


class Eval_MD():
    def __init__(self):
        super().__init__()

        self.train_model_class = ELTO_Kernel
        self.eval_model_class = ELTO_MD
        self.train_model = None
        self.eval_model = None
        self.groundtruth = None

        self.device = "cuda"
        self._is_setup = False

        self.trained_x = None

    def setup_training(self, train_obs, test_groundtruth, epochs, batch_size, window_size, d):
    
        self.train_model = self.train_model_class(train_obs, epochs, batch_size, window_size, d)
        self.trained_x = self.train_model.forward(is_deep_kernel=False)
        self.eval_model = self.eval_model_class(self.trained_x, window_size, d)

        if not isinstance(test_groundtruth, torch.tensor):
            self.groundtruth = torch.tensor(test_groundtruth, dtype=torch.complex64).to(self.device)
        else:
            self.groundtruth = test_groundtruth.to(self.device)

        self._is_setup = True

    def evaluate(self, eval_function=calculate_loss_MD, **kwargs):
        assert self.trained_x is not None
        assert self._is_setup

        estimated_eigvals = self.eval_model.compute_eigval()
        eval_mean , eval_std = eval_function(self.groundtruth, estimated_eigvals)
        # print(f'calculate mean:{eval_mean}, std:{eval_std}')
        return eval_mean, eval_std
     
def calculate_loss_MD(groundtruth, estimated_eigvals):
    _len_g = len(groundtruth)
    _len_e = len(estimated_eigvals)
    if _len_g < _len_e:
        estimated_eigvals = estimated_eigvals[:_len_g]
    elif _len_e < _len_g:
        groundtruth = groundtruth[:_len_e]
    else:
        pass
    abs_diff = torch.abs(groundtruth - estimated_eigvals)
    mean_diff = torch.mean(abs_diff).item()
    std_diff = torch.std(abs_diff).item()
    return mean_diff, std_diff