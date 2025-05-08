import torch
from torch.optim.lr_scheduler import _LRScheduler  #应该可以直接运行可以忽略报错，或尝试去掉下划线

class CustomScheduler(_LRScheduler):
    """
    define a custom learning rate scheduler for the learning rate with minimum value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        initial_learning_rate(float): Initial learning rate.
        decay_steps(int): Number of steps to decay learning rate.
        decay_rate(float): Decay rate.
        minimum_learning_rate(float): Minimum learning rate.
        last_epoch(int): The index of last epoch.(-1 represents the initial epoch).
    """

    # custom_scheduler.py
    def __init__(self, initial_learning_rate, decay_steps, decay_rate,
                 minimum_learning_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.minimum_learning_rate = minimum_learning_rate

    def get_lr(self):
        """calculate current lr"""
        current_step  = self.last_epoch
        learning_rate = self.initial_learning_rate * self.decay_rate ** (current_step / self.decay_steps)
        learning_rate = max(learning_rate, self.minimum_learning_rate)
        #return learning_rate
        return [learning_rate for _ in self.optimizer.param_groups]