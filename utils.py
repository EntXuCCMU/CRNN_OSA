import torch
import numpy as np
import random

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """
    Fixes random seeds for reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    return total_params

class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss
    stops improving or F1 score stops increasing.
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='best_model.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_score, model):
        # Assuming score is Macro-F1 (higher is better)
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            self.trace_func(f'Validation score improved ({self.best_score:.6f} --> {val_score:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)