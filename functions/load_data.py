# data select

import pandas as pd
import geopandas as gpd
import numpy as np
import os 
import re
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.model_selection import KFold
from functions.misc import *
from whittaker_eilers import WhittakerSmoother
import scipy.signal
import yaml

############################################################
############################################################

class FullModelData():

    def __init__(self, settings):

        self.settings = settings
        self.mode = settings['Mode']
        self.training_sites = self.settings['training_sites']
        self.holdout_sites = self.settings['holdout_sites']
        self.excluded_sites = self.settings['excluded_sites']

        # Hyperparameters
        self.dt = settings['forcing_freq']

        # smoothing
        self.smoothing_lambda = settings['smoothing_lambda']
        self.gap_threshold = settings['gap_threshold']

        # load in dynamic data (wave/shoreline) and static data (long/lat/orientation)        
        data_path = os.path.join(os.getcwd(),'data','model_input_data')

        self.df = pd.read_csv(os.path.join(data_path,"example_dynamic.csv"), index_col='date', parse_dates=True) 
        self.static = pd.read_parquet(os.path.join(data_path,"Regional_NSW_dataset_static.parquet"), engine="pyarrow", use_threads=True)
        
        # # define the different groups of transects...
        self.training_transects = sorted(self.get_transects(self.df, self.training_sites))
        self.holdout_transects = sorted(self.get_transects(self.df, self.holdout_sites))
        self.excluded_transects = sorted(self.get_transects(self.df, self.excluded_sites))
        self.all_transects = sorted(set(self.training_transects) | set(self.holdout_transects) | set(self.excluded_transects))

        self.QA_transects()

        self.evaluation_data = self.df.copy()
        self.train_test_split()
        self.standardize()
    
    ################################

    def get_agg_method(self, col_name):
            if col_name.endswith('_mean'):
                return 'mean'
            elif col_name.endswith('_peak'):
                return 'max'
            elif col_name.endswith('_sum'):
                return 'sum'
            else:
                return 'first'  # Default to 'first' if no suffix is present
    
    ################################

    def downscale_forcing_inputs(self):

        # Create a dictionary mapping each column to its aggregation method
        agg_methods = {col: self.get_agg_method(col) for col in self.df.columns}
        resampled_data = self.df.resample(self.dt).agg(agg_methods)

        return resampled_data

    ################################
    
    def train_test_split(self):

        self.calibration = self.df.copy()

        last_date = self.calibration.index[-1]
        dt_int = int(self.dt[:-1])

        prediction_window = self.settings['output_length'] * dt_int # 5 years in days - make sure S and L are of the dt resolution...
        lookback_window = self.settings['sequence_length'] * dt_int
        lookback_window = pd.Timedelta(lookback_window, unit='D')

        validate_start = last_date - pd.Timedelta(prediction_window*2, unit='D')
        test_start = last_date - pd.Timedelta(260 * dt_int, unit='D')
         
        # depending on whether we are in evaluation or validation mode...
        if self.mode == 'Val':
            self.train = self.calibration[:validate_start].copy()
            self.holdout = self.calibration[validate_start-lookback_window:validate_start+pd.Timedelta(prediction_window, unit='D')].copy()

        elif self.mode == 'Eval' or 'Deploy':
            self.train = self.calibration[:test_start].copy()
            self.holdout = self.calibration[test_start:].copy()


        self.whit_smooth(self.all_transects)
        self.holdout = pd.concat([self.train[test_start-lookback_window:],self.holdout], axis=0)

    ################################

    def whit_smooth(self, profiles):

        for ThisP in profiles:

            self.remove_outliers(ThisP)

            interpolated_series = self.train[ThisP].interpolate(limit=self.gap_threshold)
            profile_time_series = self.train[ThisP].to_numpy(copy=True)

            weights = np.where(np.isnan(profile_time_series), 0, 1)  # 0 for NaNs, 1 for real data
            profile_time_series[np.isnan(profile_time_series)] = -9999 # NaN's cannot be allowed to propogate through the smoothing function... 
            
            whittaker_smoother = WhittakerSmoother(
                                        lmbda = self.smoothing_lambda,
                                        order=2, 
                                        data_length=len(profile_time_series), 
                                        weights=weights)
            
            smoothed_time_series = whittaker_smoother.smooth(profile_time_series)
            self.train[ThisP] = smoothed_time_series

            nan_indices = interpolated_series[interpolated_series.isna()].index
            self.train.loc[nan_indices, ThisP] = np.nan

    ################################

    def remove_outliers(self, ThisP):

        mu = self.calibration[ThisP].mean()
        std = self.calibration[ThisP].std()

        z_threshold = 3
        z_scores = (self.calibration[ThisP] - mu) / std
        self.calibration.loc[np.abs(z_scores) >= z_threshold, ThisP] = np.nan

    ################################
    def standardize(self):

        # --- Load precomputed transformation parameters ---
        with open("config/transformation_parameters.yml", "r") as f:
            global_transformation_params = yaml.safe_load(f)

        # Store parameters in the same style you want
        self.transformation_parameters = (
            global_transformation_params['target']['mu'],
            global_transformation_params['target']['std']
        )
        self.dynamic_transformation_parameters = {
            k: v for k, v in global_transformation_params.items() if k != "target"
        }

        # --- Apply to data ---
        all_transects = list(set(self.training_transects + self.holdout_transects))

        # Standardise target
        target_mu, target_sigma = self.transformation_parameters
        target_train = (self.train[all_transects] - target_mu) / target_sigma
        target_holdout = (self.holdout[all_transects] - target_mu) / target_sigma

        training_varList = [target_train, self.train[['sin_t', 'cos_t']]]
        holdout_varList = [target_holdout, self.holdout[['sin_t', 'cos_t']]]

        # Standardise dynamic variables with precomputed params
        for var in self.settings['dynamic_inputs']:
            Tvar_columns = [name + '_' + var for name in all_transects]

            params = self.dynamic_transformation_parameters[var]
            Tvar_mu = params['mu']
            Tvar_sigma = params['sigma']

            Tvar_matrix = (self.train[Tvar_columns] - Tvar_mu) / Tvar_sigma
            holdout_matrix = (self.holdout[Tvar_columns] - Tvar_mu) / Tvar_sigma

            training_varList.append(Tvar_matrix)
            holdout_varList.append(holdout_matrix)

        # --- Final concat ---
        self.train = pd.concat(training_varList, axis=1)
        self.holdout = pd.concat(holdout_varList, axis=1)

        ##########
        # Standardize static data
        ##########
        exclude_from_standardization = ['Relative_Transect_Position']

        # Standardize `static`, skipping categorical variables and excluded variables
        if self.settings.get('catagorical_static_inputs'):  
            cols_to_standardize_static = self.static.columns.difference(self.settings['catagorical_static_inputs'] + exclude_from_standardization)
        else:
            cols_to_standardize_static = self.static.columns.difference(exclude_from_standardization)

        temp_mean_static = self.static[cols_to_standardize_static].mean()
        temp_std_static = self.static[cols_to_standardize_static].std()

        self.static[cols_to_standardize_static] = (self.static[cols_to_standardize_static] - temp_mean_static) / temp_std_static

    ################################

    def get_transects(self, df, beach_names):

        # Define regex pattern for 'ausXXXX-XXXX' - this will have to be more flexible if we bring
        # other training sites... but does for now
        pattern = re.compile(r'aus\d{4}-\d{4}')
        # Find matching columns

        matching_columns = []

        for col in df.columns:
             if pattern.fullmatch(col) and (col.split('-')[0] in beach_names):
                matching_columns.append(col)

        return matching_columns
    
    ################################
    
    def QA_transects(self):

        # Remove invalid transects from training and evaulation...
        # Those that go through river mouths, headlands etc...

        invalid_transect_data = gpd.read_file('data/invalid_transects.geojson')
        invalid_transects = invalid_transect_data['TransectId'].tolist()

        self.training_transects = [item for item in self.training_transects if item not in invalid_transects]
        self.holdout_transects = [item for item in self.holdout_transects if item not in invalid_transects]
        self.excluded_transects = [item for item in self.excluded_transects if item not in invalid_transects]
        self.all_transects = [item for item in self.all_transects if item not in invalid_transects]

    ################################

