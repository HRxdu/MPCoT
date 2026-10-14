import numpy as np
import torch
import pickle
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model,feature_masks,metapath_masks,metapath_mask_logit_result,result_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,feature_masks,metapath_masks,metapath_mask_logit_result,result_dir)
        elif score <= self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,feature_masks,metapath_masks,metapath_mask_logit_result,result_dir)
            self.counter = 0
    """
    def save_checkpoint(self, val_loss, model,feature_masks,metapath_masks,metapath_mask_logit_result,result_dir):
        if self.verbose:
            print(f'Validation result increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        
        if not os.path.isdir(self.save_path+f'/{result_dir}'):
            os.mkdir(self.save_path+f'/{result_dir}')
        if not os.path.isdir(f'D:/Pycharm_project/interpretable_TOA/static_mode/Explainer/output/mask_logit_result/{result_dir}'):
            os.mkdir(f'D:/Pycharm_project/interpretable_TOA/static_mode/Explainer/output/mask_logit_result/{result_dir}')

        torch.save(model.state_dict(), self.save_path+f'/{result_dir}/checkpoint.pt')

        with open(f'output/model/{result_dir}/feature_masks.pkl', 'wb') as f:
            pickle.dump(feature_masks, f)
        with open(f'output/model/{result_dir}/metapath_masks.pkl', 'wb') as f:
            pickle.dump(metapath_masks, f)

        with open(f'D:/Pycharm_project/interpretable_TOA/static_mode/Explainer/output/mask_logit_result/{result_dir}/metapath_mask_logit_result.pkl', 'wb') as file:
            pickle.dump(metapath_mask_logit_result, file)

        self.val_loss_min = val_loss
        """

    def save_checkpoint(self, val_loss, model, feature_masks, metapath_masks, metapath_mask_logit_result, result_dir):
        if self.verbose:
            print(f'Validation result increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with open(f'{result_dir}/metapath_masks.pkl', 'wb') as f:
            pickle.dump(metapath_masks, f)

        with open(
                f'{result_dir}/metapath_mask_logit_result.pkl',
                'wb') as file:
            pickle.dump(metapath_mask_logit_result, file)

        self.val_loss_min = val_loss