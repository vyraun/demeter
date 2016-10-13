import numpy as np

class LinearAnnealer(object):

    def __init__(self, init_exp,
                       final_exp,
                       anneal_steps):

        self.init_exp = init_exp
        self.final_exp = final_exp
        self.anneal_steps = anneal_steps

        self.exp = init_exp

    def anneal(self, train_iter):
        if self.anneal_steps == 0: 
            self.exp = 0.0
        else:
            ratio = max((self.anneal_steps - train_iter) / float(self.anneal_steps), 0)
            self.exp = (self.init_exp - self.final_exp) * ratio + self.final_exp
    
    def is_explore_step(self):
        if np.random.random() < self.exp:
            return True
        else:
            False
