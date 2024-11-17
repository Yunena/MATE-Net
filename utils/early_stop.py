import numpy as np


class EarlyStop:
    def __init__(self,stop_epoch = 10,delta=-0.01):
        self.train_biacc = 0.93
        self.biacc = 0.
        self.train_auc = 0.93
        self.auc = 0.
        self.current_epoch = None
        self.patience = stop_epoch
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.counter = 0

    def check_stop(self,val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            return True

    def attn_check_stop(self,val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            return True

    def initialize(self):
        self.biacc = 0.
        self.auc = 0.
        self.current_epoch = None
        self.best_score = None
        self.early_stop = False
        self.counter = 0

