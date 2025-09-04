import os
import numpy as np
import pandas as pd
import copy
import torch
import re
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from functions.custom_datasets import *
from functions.custom_loss_functions import *
from functions.model import TFT
import matplotlib.ticker as ticker
from functions.misc import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Transformer25:

    def __init__(self, data, settings, CVFold = None):
        """
        Initialize model settings, device, and data.

        Inputs:
            - training_data: dataframe containing training samples, index in datetime
            - holdout_data: dataframe containing validation / unseen samples, depending on the context of this object
            - settings: dictionary containing hyperparameters and model config options

        Output
            - model: trained pytorch model,
            - nmse: Normalized mean square error on holdout data
        """
        self.ground_truth_obs = data.evaluation_data
        self.settings = settings
        self.mode = settings['Mode']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()  # <-- NEW: AMP Gradient Scaler
        self.nan_token = -9999

        #bring in static data
        self.static_data = data.static
        
        # this is data that is seperated in time...
        self.training_data = data.train
        self.holdout_data = data.holdout

        if CVFold == None:
            self.target = data.training_transects
            self.holdout_target = data.holdout_transects
        else:
            self.target = CVFold[0]
            self.holdout_target = CVFold[1]

        # transformation parameters for eval
        self.transformation_parameters = data.transformation_parameters
        
        # Hyperparameters
        self.batch_size = self.settings['batch_size']
        self.sequence_length = self.settings['sequence_length']
        self.output_length = self.settings['output_length']
        self.lr = float(self.settings['learning_rate'])
        self.warmup = settings['warmup']
        self.epochs = self.settings['epochs']
        self.num_workers = 1  # Fixed for single-threaded processing
        
        # Regularization
        # self.early_stop = self.settings['early_stop']
        self.continuity_lambda = self.settings['init_lambda']
        self.variance_lambda = self.settings['variance_lambda']
        self.l1_lambda = self.settings['l1_lambda']
        self.l2_lambda = self.settings['l2_lambda']
        self.trend_lambda = self.settings['trend_lambda']

        self.seed = self.settings['seed']

    ############
    def run(self):

        """
        The datalaoders will be built, model trained and evaluated on the holdout_data
        """
        
        self.build_dataloaders()
        self.build_model()
        self.train_model()

        RoS = self.evaluate_model()
    
        return self.model, RoS

    ############

    def build_dataloaders(self):
        """
        Set up Datasets,
        for training, we are gather data from all transects (specified in self.targets) to generate a generalized model.
        We do the same for evaluation but we do not shuffle to maintain order...
        """
        training_datasets = []
        unseen_datasets = []

        # self.eval_transects = sorted(list(set(self.target) | set(self.holdout_target)))


        for Thistarget in self.target:
            This_training_dataset = seq2seqDataset(self.training_data, Thistarget, self.static_data, self.settings)
            training_datasets.append(This_training_dataset)

        for Thistarget in self.holdout_target:
            This_unseen_dataset = InferenceDataset(self.holdout_data,  Thistarget, self.static_data, self.settings)
            unseen_datasets.append(This_unseen_dataset)

        training_dataset = ConcatDataset(training_datasets)
        holdout_dataset = ConcatDataset(unseen_datasets)

        """
        Data Loader for Training
        """
        use_pin_memory = self.settings['Mode'] != 'Val'
        g = torch.Generator()
        g.manual_seed(self.seed)

        self.training_loader = DataLoader(training_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True, # we want to shuffle the sequences, so there is not bias towards earlier beaches (or more northern beaches as that is how the beaches are orderd) 
                                  num_workers=self.num_workers, 
                                  pin_memory= use_pin_memory,
                                  generator=g)
        
        """
        Data Loaders for Validation / Evaluation
        """
        self.holdout_loader = DataLoader(holdout_dataset, 
                                         batch_size=self.batch_size, 
                                         shuffle=False, # maintain order so we know which prediction belongs to which transect
                                         drop_last=False,
                                         pin_memory= use_pin_memory)
        
    ############
        
    def build_model(self, model):
        """
        Initialize model, optimizer, scheduler, and loss function.
        """
        if model == None:
            self.model = TFT(settings = self.settings)
            self.model.to(self.device)
        else:
            self.model = model
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_function = quantile_loss

        # warm up and decay scheduler
        warmup_limit = round(self.settings['epochs'] * self.warmup)
        warmup_scheduler = lr_scheduler.LinearLR(
                                        self.optimizer, 
                                        start_factor=0.1,
                                        end_factor=1.0,
                                        total_iters=warmup_limit
                                        )
        decay_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                         T_max=self.epochs-warmup_limit)

        self.scheduler = lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_limit])

    ############

    def train_model(self):
        """
        Run the main training loop over all epochs.
        """
        progress_bar = tqdm(total=self.epochs, desc="Training Progress", unit="epoch")

        for self.thisEpoch in np.arange(self.epochs):

            training_loss = self.train_step()
            current_lr = self.scheduler.get_last_lr()[0]

            progress_bar.set_description(f"Loss: {training_loss:.3f}, LR: {current_lr:.6f}")
            progress_bar.update()
            self.scheduler.step()

        progress_bar.close()
    
    ############
    
    def train_step(self):
        """
        Execute a single training step (batch-level training) and update model weights. there are a couple of reg terms i was trialing here, but ended up setting them to zero.
        """

        self.model.train()
        num_batches = len(self.training_loader)
        total_loss = 0

        for _, (Xp, Xf, Xs, y, target) in enumerate(self.training_loader):
            Xs_num, Xs_cat = Xs

            Xp = Xp.to(self.device)
            Xf = Xf.to(self.device)
            Xs_num = Xs_num.to(self.device)
            Xs_cat = Xs_cat.to(self.device)
            y = y.to(self.device)

            # ----> Quantile Loss <----
            preds = self.model(Xp, Xf, Xs_num, Xs_cat)
            q_loss = self.loss_function(preds, y)

            # ----> Trend Matching Loss <----
            trend_true = y[:, -1] - y[:, 0]  # Net shoreline change in true data
            trend_pred = preds[:, -1, 1] - preds[:, 0, 1]  # Net change in predicted shoreline

            valid_trend_mask = (y[:, 0] != self.nan_token) & (y[:, -1] != self.nan_token)
            trend_loss = torch.mean((trend_true[valid_trend_mask] - trend_pred[valid_trend_mask]) ** 2)

            # ----> Continuity Penalty <----
            prev_y = Xp[:, -1, -1]
            valid_mask = prev_y != self.nan_token

            valid_preds = preds[:, 0, :][valid_mask]
            valid_prev_y = prev_y[valid_mask].unsqueeze(-1)

            continuity_loss = torch.mean((valid_preds - valid_prev_y) ** 2)

            # ----> Variance Penalty <----
            y_zerod = torch.where(y == -9999, 0, y)

            var_y = torch.var(y_zerod, dim=1, unbiased=False)  # Variance along time axis (L)
            var_preds = torch.var(preds[:, :, 1], dim=1, unbiased=False)

            variance_loss = torch.mean((var_y - var_preds) ** 2)

            # ----> Elastic Net Regularization <----
            l1_loss = torch.tensor(0., device=self.device)
            l2_loss = torch.tensor(0., device=self.device)

            for param in self.model.parameters():
                l1_loss += torch.norm(param, 1)
                l2_loss += torch.norm(param, 2) ** 2

            elastic_loss = self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

            # ----> Compute Total Loss  <----
            loss = (
                q_loss
                + elastic_loss
                + self.continuity_lambda * continuity_loss
                + self.variance_lambda * variance_loss
                + self.trend_lambda * trend_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item()

        avg_loss = total_loss / num_batches
        return avg_loss
    
    ###########

    def evaluate_model(self):
        """
        Evaluates the model using Quantile Loss (Pinball Loss).
        """
        self.model.eval()
        model_output = predict(self.holdout_loader, self.model)
        obs = torch.tensor(self.holdout_data[self.holdout_target][-1 * (self.settings['output_length']):].to_numpy().T).to(self.device) # transpose to ensure output and obs are both of shape '# of transects' x 'prediction length' 
        qloss = self.loss_function(model_output, y)

        return qloss

    #####################################################

    def get_transects(self, df, beach_names):

        # Define regex pattern for 'ausXXXX-XXXX' - this will have to be more flexible if we bring
        # foregin training sites... but does for now
        pattern = re.compile(r'aus\d{4}-\d{4}')
        # Find matching columns
        matching_columns = [
            col for col in df.columns 
            if pattern.fullmatch(col) and any(beach in col for beach in beach_names)
        ]
        return matching_columns

#####################################################
#####################################################
#####################################################
     
def predict(data_loader, model):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########
    '''Output Init for Traditional Structure'''
    ########

    outputs = []
    
    with torch.no_grad():
 
        for ii, (Xp, Xf, Xs, _) in enumerate(data_loader):

            Xs_num, Xs_cat = Xs
            
            Xp = Xp.to(device)
            Xf = Xf.to(device)
            Xs_num = Xs_num.to(device)
            Xs_cat = Xs_cat.to(device)

            model_output = model(Xp, Xf, Xs_num, Xs_cat)
            outputs.append(model_output)

    return torch.cat(outputs, dim=0)

#####################################################
#####################################################
#####################################################


def generate_output(target, model, data, settings, ablation_type = None):

    ground_truth = data.df.copy()
    for ThisP in target:

        mu = ground_truth[ThisP].mean()
        std = ground_truth[ThisP].std()

        z_threshold = 3
        z_scores = (ground_truth[ThisP] - mu) / std
        ground_truth.loc[np.abs(z_scores) >= z_threshold, ThisP] = np.nan

    mu = data.transformation_parameters[0]
    sigma = data.transformation_parameters[1]

    unseen_datasets = []

    for Thistarget in target:
        This_unseen_dataset = InferenceDataset(data.holdout, Thistarget, data.static,  settings, ablation_type)
        unseen_datasets.append(This_unseen_dataset)

    unseen_dataset = ConcatDataset(unseen_datasets)
    unseen_loader = DataLoader(unseen_dataset, batch_size=settings['batch_size'], shuffle=False, drop_last=False)

    model_output = predict(unseen_loader, model).cpu().numpy()
    obs = ground_truth[target][-1 * (settings['output_length']):].to_numpy().T
    # obs = data.holdout[unseen_target][-1 * (settings['output_length']):].to_numpy().T
    # obs = (obs*sigma)+mu

    nmse_list_perTransect = []
    rmse_list_perTransect = []
    r2_list_perTransect = []
    mse_list_perTransect = []
    mae_list_perTransect = []
    qloss_list_perTransect = []

    output = {}

    for ii, transect in enumerate(target):

        This_model_output = (model_output[ii] * sigma) + mu

        # y0 = data.calibration[transect][-settings['output_length']:][0]

        # # detrending
        # m, _ = data.trend_params[transect]
        # if np.isnan(m):
        #     m=0

        # T = This_model_output.shape[0]
        # trend_line = m * np.arange(T) # shape: [T]
        # # # Add the trend to each quantile
        # This_model_output = This_model_output + trend_line[:, None]

        ThisPred = This_model_output[:, 1]
        ThisObs = obs[ii, :]

        # Updated: Get six metrics
        rmse, nmse, r2, mse, mae, mbe = calculate_skill(ThisObs, ThisPred)

        # Append to corresponding lists
        nmse_list_perTransect.append(nmse)
        rmse_list_perTransect.append(rmse)
        r2_list_perTransect.append(r2)
        mse_list_perTransect.append(mse)
        mae_list_perTransect.append(mae)
        qloss_list_perTransect.append(mbe)

        # Save full model output per transect
        output[target[ii]] = This_model_output

    # Load static data for scores
    data_path = os.path.join(os.getcwd(), 'data', 'model_input_data')
    static_data = pd.read_parquet(
        os.path.join(data_path, "Regional_NSW_dataset_static.parquet"),
        engine="pyarrow",
        use_threads=True
    )

    # Build the scores DataFrame
    scores = static_data.loc[target].copy()

    scores['nmse'] = nmse_list_perTransect
    scores['rmse'] = rmse_list_perTransect
    scores['r2'] = r2_list_perTransect
    scores['mse'] = mse_list_perTransect
    scores['mae'] = mae_list_perTransect
    scores['q50'] = qloss_list_perTransect

    return output, scores

##########################
##########################




    

    





    



    