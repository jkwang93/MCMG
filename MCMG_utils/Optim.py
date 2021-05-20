# _*_ encoding: utf-8 _*_
__author__ = 'wjk'
__date__ = '2020/6/18 22:10'

'''A wrapper class for optimizer '''
# import matplotlib.pyplot as plt
import numpy as np


# From https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        # print(lr)

# opts = [ScheduledOptim(64, 1, 200),
#         ScheduledOptim(512, 1, 8000),
#         ScheduledOptim(256, 1, 4000)]
# plt.plot(np.arange(1, 20000), [[opt._get_lr_scale for opt in opts] for i in range(1, 20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])
# plt.show()