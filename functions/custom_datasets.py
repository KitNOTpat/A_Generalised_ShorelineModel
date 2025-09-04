# implement custom data loaders for LSTM, linear and non-linear probe

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import warnings

class seq2seqDataset(Dataset):
    def __init__(self, dataframe, target, static, settings):

        self.nan_token = -9999
        self.target = target

        if settings['dynamic_inputs'] != None:
            self.dynamic_inputs = [f"{self.target}_" + s for s in settings['dynamic_inputs']]
        else:
            self.dynamic_inputs = []
        
        self.dynamic_inputs = self.dynamic_inputs + ['sin_t', 'cos_t']

        self.static_numerical = settings['numerical_static_inputs']
        self.static_catagorical = settings['catagorical_static_inputs']

        self.historical_inputs = self.dynamic_inputs
        if settings['use_prev_target']:
            self.historical_inputs = self.historical_inputs + [self.target]

        self.sequence_length = settings['sequence_length']
        self.output_length = settings['output_length']
        self.noise_std = 0.1 # 0.1 # add noise to target to imporve robustness...
        # self.noise_std = 2 # 0.1 # add noise to target to imporve robustness...

        self.historical = torch.tensor(dataframe[self.historical_inputs].to_numpy(copy=True)).float()
        self.future = torch.tensor(dataframe[self.dynamic_inputs].to_numpy(copy=True)).float()
        self.y = torch.tensor(dataframe[self.target].to_numpy(copy=True)).float()

        self.static_numerical = torch.tensor(static.loc[self.target][self.static_numerical].to_numpy(copy=True)).float()

        try:
            self.static_catagorical = torch.tensor(
                static.loc[self.target][self.static_catagorical].to_numpy(copy=True)
            ).float()
        except KeyError:  # Handles the case where self.static_catagorical is empty or invalid
            self.static_catagorical = torch.empty((0)).float()  # Empty tensor
    
    def __len__(self):

        length = math.ceil((self.y.shape[0] - self.sequence_length)/ (self.output_length+52))
 
        return length
    
    def __getitem__(self, i):

        i = (self.output_length+self.sequence_length) + i*(self.output_length+52)

        pred_start = i - self.output_length
        sequence_start = i - self.output_length - self.sequence_length

        if i < self.y.shape[0]:

            x_past = self.historical[ sequence_start : pred_start]
            x_future = self.future[ pred_start : i] 
            y = self.y[ pred_start : i]
        
        else:

            # check to see if there are any observations in the last sequence on each transect..
            if pred_start > self.y.shape[0]: 
                raise ValueError(
                    f'Last sequence along Transect begins prediction at index {pred_start}, but last valid observation ends at index {self.y.shape[0]}. No observations to train on. Modify sequence & output length.')

            else:

                x_past = self.historical[ sequence_start : pred_start]

                # we need to add some padding on the x_future and y tensors to ensure consistent L Dimension (output_length)
                residual_valid = (self.y.shape[0] - pred_start)
                padding =  self.output_length - residual_valid

                x_future = self.future[ pred_start : i]
                y = self.y[ pred_start : i]

                y_padding_tensor = torch.full((padding,1), float('nan'))
                x_future_padding_tensor = torch.full((padding,len(self.dynamic_inputs)), float('nan'))

                y = torch.cat((y, torch.squeeze(y_padding_tensor, dim=-1)), axis = 0)
                x_future = torch.cat((x_future, torch.squeeze(x_future_padding_tensor, dim=-1)), axis = 0)

        ############
        ############
        # i = (i+1) * (self.output_length+self.sequence_length)

        # pred_start = i - self.output_length
        # sequence_start = i - self.output_length - self.sequence_length

        # if i < self.y.shape[0]:

        #     x_past = self.historical[ sequence_start : pred_start]
        #     x_future = self.future[ pred_start : i] 
        #     y = self.y[ pred_start : i]
        
        # else:

        #     # check to see if there are any observations in the last sequence on each transect..
        #     if pred_start > self.y.shape[0]: 
        #         raise ValueError(
        #             f'Last sequence along Transect begins prediction at index {pred_start}, but last valid observation ends at index {self.y.shape[0]}. No observations to train on. Modify sequence & output length.')

        #     else:

        #         x_past = self.historical[ sequence_start : pred_start]

        #         # we need to add some padding on the x_future and y tensors to ensure consistent L Dimension (output_length)
        #         residual_valid = (self.y.shape[0] - pred_start)
        #         padding =  self.output_length - residual_valid

        #         x_future = self.future[ pred_start : i]
        #         y = self.y[ pred_start : i]

        #         y_padding_tensor = torch.full((padding,1), float('nan'))
        #         x_future_padding_tensor = torch.full((padding,len(self.dynamic_inputs)), float('nan'))

        #         y = torch.cat((y, torch.squeeze(y_padding_tensor, dim=-1)), axis = 0)
        #         x_future = torch.cat((x_future, torch.squeeze(x_future_padding_tensor, dim=-1)), axis = 0)


        ############
        ############


        # add some noise to the past and future observations to improve generalization...
        y = y + torch.randn_like(y) * self.noise_std
        x_past[:, -1] += torch.randn_like(x_past[:, -1]) * self.noise_std
        # x_past += torch.randn_like(x_past) * self.noise_std
        # x_future += torch.randn_like(x_future) * self.noise_std

        # !!!!!!!!!!
        # we need to do a critical check, In attention we use a causal mask, and we also have a mask for NaN'd values of the
        # past target (transformed to -9999). However, if the first value of x_past is -9999, it will mask itself, and all 
        # future timesteps, meaning no attention can take place and it will return NaN output... to ensure all timesteps have at least
        # one timestep to attend too (even if its just itself), we replace the first -9999, if it exists with the mean past target value... 

        if torch.isnan(x_past[0,-1]):
            valid_past_target = x_past[:,-1][~torch.isnan(x_past[:,-1])]
            x_past[0,-1] = torch.mean(valid_past_target)

        x_past = torch.nan_to_num(x_past, nan=self.nan_token)
        x_future = torch.nan_to_num(x_future, nan=self.nan_token)
        y = torch.nan_to_num(y, nan=self.nan_token)
        
        return x_past, x_future, (self.static_numerical, self.static_catagorical), y, self.target

 
        
#####################################################
#####################################################
    
class InferenceDataset(Dataset):
    def __init__(self, dataframe, target, static, settings, ablation_type = None):
        
        self.nan_token = -9999
        self.target = target
        self.ablation_type = ablation_type

        if settings['dynamic_inputs'] != None:
            self.dynamic_inputs = [f"{self.target}_" + s for s in settings['dynamic_inputs']]
        else:
            self.dynamic_inputs = []
        self.dynamic_inputs = self.dynamic_inputs + ['sin_t', 'cos_t']
   
        self.static_numerical_inputs = settings['numerical_static_inputs']
        self.static_catagorical = settings['catagorical_static_inputs']

        self.historical_inputs = self.dynamic_inputs
        if settings['use_prev_target']:
            self.historical_inputs = self.historical_inputs +[self.target]

        self.sequence_length = settings['sequence_length']
        self.output_length = settings['output_length']

        self.historical = torch.tensor(dataframe[self.historical_inputs].to_numpy()).float()
        self.future = torch.tensor(dataframe[self.dynamic_inputs].to_numpy()).float()
        self.y = torch.tensor(dataframe[self.target].to_numpy()).float()  

        self.static_numerical = torch.tensor(static.loc[self.target][self.static_numerical_inputs].to_numpy()).float()

        try:
            self.static_catagorical = torch.tensor(
                static.loc[self.target][self.static_catagorical].to_numpy()
            ).float()
        except KeyError:  # Handles the case where self.static_catagorical is empty or invalid
            self.static_catagorical = torch.empty((0)).float()  # Empty tensor

    def __len__(self):
        
        return 1

    def __getitem__(self, i):

        pred_start = -1 * self.output_length
        sequence_start =  -1 * (self.output_length + self.sequence_length)

        x_past = self.historical[ sequence_start : pred_start]
        x_future = self.future[ pred_start :] 
        y = self.y[ pred_start :]

        # !!!!!!!!!!
        # we need to do a critical check, In attention we use a causal mask, and we also have a mask for NaN'd values of the
        # past target (transformed to -9999). However, if the first value of x_past is -9999, it will mask itself, and all 
        # future timesteps, meaning no attention can take place and it will return NaN output... to ensure all timesteps have at least
        # one timestep to attend too (even if its just itself), we replace the first -9999, if it exists with the mean past target value... 

        if torch.isnan(x_past[0,-1]):
            valid_past_target = x_past[:,-1][~torch.isnan(x_past[:,-1])]
            x_past[0,-1] = torch.mean(valid_past_target)

        x_past = torch.nan_to_num(x_past, nan=self.nan_token)
        x_future = torch.nan_to_num(x_future, nan=self.nan_token)
        y = torch.nan_to_num(y, nan=self.nan_token)

        # Ablation tests
        if self.ablation_type is not None:

            if self.ablation_type == 'static':
                self.static_numerical = torch.randn_like(self.static_numerical)*2

            elif self.ablation_type == 'signal':
                x_past[:, -1] = torch.randn_like(x_past[:, -1])*2

            elif self.ablation_type == 'signal_partial':
                # Keep the last 10 timesteps intact
                keep_n = 8
                noise = torch.randn_like(x_past[:-keep_n, -1])*2
                x_past[:-keep_n, -1] = noise

            elif self.ablation_type == 'dynamic':
                x_past[:, :-1] = torch.randn_like(x_past[:, :-1])*2  # zero dynamic inputs (excluding prev_target)
                x_future = torch.randn_like(x_future)*2

            else:
                # Specific static variable ablation by name (e.g., 'beach_length')
                # i = self.static_numerical_inputs.index(self.ablation_type)
                # self.static_numerical[i] = np.random.normal(0, 1)


                i = self.static_numerical_inputs.index(self.ablation_type)
                z = self.static_numerical_inputs.index('Relative_Transect_Position')

                for j in range(len(self.static_numerical_inputs)):
                    if j not in (i, z):
                        self.static_numerical[j] = np.random.normal(0, 1)


                # for j in range(len(self.static_numerical_inputs)):
                #     if j != self.static_numerical_inputs.index(self.ablation_type):
                #         self.static_numerical[j] = np.random.normal(0, 1)

        return x_past, x_future, (self.static_numerical,self.static_catagorical), y

#####################################################
#####################################################