import numpy as np
import cma
from matplotlib import pyplot as plt
from scipy import stats
from .train_eval import Eval_pendulum, Eval_MD, calculate_loss_MD

# dataset:
# pendulum simulation reference: the kernel Kalman rule

# regularization parameter for the inverses
eps_t = -6
eps_o = -6
eps_q = -8

epochs = 200
batch_size = 50
window_size = 5
d = window_size

elto_experiment = Eval_pendulum()

if elto_experiment == Eval_pendulum():
    elto_experiment.setup_training(train_data, test_obs, test_groundtruth, epochs, batch_size, window_size, d)

    parameter_names = ['eps_t', 'eps_o', 'eps_q']
    # , 'bandwidth_factor_k', 'bandwidth_factor_g'

    @parameter_naming(parameter_names)
    @parameter_transform(np.exp)
    @exception_catcher(np.linalg.linalg.LinAlgError, 1e4)

    def eval_experiment(**kwargs):
        return elto_experiment.evaluate(**kwargs)

    x_0 = [eps_t, eps_o, eps_q]

    cma_opt = cma.CMAEvolutionStrategy(x_0, 0.5)
    cma_opt.optimize(objective_fct=eval_experiment, iterations=200, verb_disp=1)

elif elto_experiment == Eval_MD():
    elto_experiment.setup_training(train_data, test_groundtruth, epochs, batch_size, window_size, d)
    elto_experiment.evaluate(eval_function=calculate_loss_MD)