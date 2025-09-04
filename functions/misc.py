# miscellaneous functions

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection
import matplotlib.colors
import torch
from matplotlib.colors import LinearSegmentedColormap
import random
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans

#####################################################
#####################################################

# def calculate_skill(obs, preds):

#     obs_non_nan = obs[~np.isnan(obs)]
#     preds_non_nan = preds[~np.isnan(obs)]

#     mse = mean_squared_error(obs_non_nan, preds_non_nan)
#     rmse = np.sqrt(mse)    
#     true_variance = np.var(obs_non_nan)
#     nmse = mse / true_variance
#     # nmse = np.sqrt(mse)
    
#     r2 = r2_score(obs_non_nan, preds_non_nan)

#     return rmse, nmse, r2


def calculate_skill(obs, preds):
    """
    Computes RMSE, NMSE, R², MSE, MAE, and median quantile loss (pinball loss)
    for given observations and predictions. Ignores NaNs in obs when computing metrics.
    """
    # Remove NaNs from obs and align preds
    obs_non_nan = obs[~np.isnan(obs)]
    preds_non_nan = preds[~np.isnan(obs)]

    # Compute MSE and RMSE
    mse = mean_squared_error(obs_non_nan, preds_non_nan)
    rmse = np.sqrt(mse)

    # Compute Normalized MSE (NMSE)
    true_variance = np.var(obs_non_nan)
    nmse = mse / true_variance if true_variance > 0 else np.nan

    # Compute R² Score
    r2 = r2_score(obs_non_nan, preds_non_nan)

    # Compute MAE (Mean Absolute Error)
    mae = mean_absolute_error(obs_non_nan, preds_non_nan)

    # Compute Pinball Loss for quantile = 0.5 (median)
    q = 0.5
    errors = obs_non_nan - preds_non_nan
    pinball = np.where(errors > 0, q * errors, (q - 1) * errors)
    qloss = np.mean(pinball)

    return rmse, nmse, r2, mse, mae, qloss

# def calculate_skill(obs, preds):
#     """
#     Computes RMSE, NMSE, R², MBE, MAE, and median quantile loss (pinball loss)
#     for given observations and predictions. Ignores NaNs in obs when computing metrics.
#     """
#     # Remove NaNs from obs and align preds
#     obs_non_nan = obs[~np.isnan(obs)]
#     preds_non_nan = preds[~np.isnan(obs)]

#     # Compute MSE and RMSE
#     mse = mean_squared_error(obs_non_nan, preds_non_nan)
#     rmse = np.sqrt(mse)

#     # Compute Normalized MSE (NMSE)
#     true_variance = np.var(obs_non_nan)
#     nmse = mse / true_variance if true_variance > 0 else np.nan

#     # Compute R² Score
#     r2 = r2_score(obs_non_nan, preds_non_nan)

#     # Compute MAE (Mean Absolute Error)
#     mae = mean_absolute_error(obs_non_nan, preds_non_nan)

#     # Compute MBE (Mean Bias Error)
#     mbe = np.mean(preds_non_nan - obs_non_nan)

#     # Compute Pinball Loss for quantile = 0.5 (median)
#     q = 0.5
#     errors = obs_non_nan - preds_non_nan
#     pinball = np.where(errors > 0, q * errors, (q - 1) * errors)
#     qloss = np.mean(pinball)

#     return rmse, nmse, r2, mbe, mae, qloss

#####################################################
#####################################################

def brier_skill_score(preds, base, obs):

    obs_non_nan = obs[~np.isnan(obs)]
    preds_non_nan = preds[~np.isnan(obs)]
    base_non_nan = base[~np.isnan(obs)]

    mse = mean_squared_error(obs_non_nan, preds_non_nan)   
    base_mse = mean_squared_error(obs_non_nan, base_non_nan) 

    bss = 1 - (mse/base_mse)  
 
    return bss

#####################################################
#####################################################

def calculate_power(H,T):
    Power = ((1025 * (9.81**2) * (H**2) * (T**2))/(8*pi))
    return Power/1e6

#####################################################
#####################################################

def calculate_energy(H):

    delta = H.index.to_series().diff()[1]
    dt = delta.total_seconds() / 3600
    # br 
    E = ((1/16) * 1025 * 9.81 * dt * (H**2).sum(axis=1))/1e6
    # E = ((1/16) * 1025 * 9.81 * dt * (H**2))/1e6

    return E

#####################################################
#####################################################

def destandardize(arr, scalers, target):
    _arr_ = (arr*scalers[f'{target}_std'])+scalers[f'{target}_m']
    return _arr_

#####################################################
#####################################################

def standardize(arr, scalers, target):
    _arr_ = (arr - scalers[f'{target}_m'])/scalers[f'{target}_std']
    return _arr_

#####################################################
#####################################################

def stdize(df):
    std_df = (df - df.mean(axis=0))/df.std()
    return std_df

#####################################################
#####################################################

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

#####################################################
#####################################################

def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

#####################################################
#####################################################

def plot_timeseries(data):

    fig, ax = plt.subplots(figsize=(12,4),facecolor='white')
    fig.tight_layout(pad=5.0)
    ax.grid(linestyle = '--', linewidth = 1, axis='both')

    ax.plot(data.index, data, c = 'mediumvioletred')

#####################################################
#####################################################

def plot_train_test(data, settings):
    fig, axs = plt.subplots(len(settings['target']),figsize=(15,len(settings['target'])*4),facecolor='white')
    axs = np.atleast_1d(axs)

    for ii, ax in enumerate(axs):
        ax.set_title(settings['target'][ii])
        ax.grid(linestyle = '--', linewidth = 1, axis='both')

        ax.scatter(data.train.index, data.train[settings['target'][ii]], color = 'royalblue', facecolor = 'w', alpha = 1, s = 30, zorder = 2, marker = 's')
        ax.scatter(data.test.index, data.test[settings['target'][ii]], color = 'crimson', facecolor = 'w', alpha = 1, s = 30, zorder = 2, marker = 's')
        ax.axvspan(data.train.index[0], data.train.index[-1], color='grey', alpha=.15, lw=1, zorder = 0, fill=True , label = 'Calibration Data')
        ax.plot(data.df[settings['target'][ii]].dropna().index, data.df[settings['target'][ii]].dropna(), color = 'grey', linewidth = 1, alpha = 0.6, zorder = 0)
    
#####################################################
#####################################################

def plot_shorelines(data, settings):

    interpolated_shorelines = data.df[settings['target']][data.first_valid_index:data.last_valid_index].interpolate(method='linear')
    interpolated_shorelines.index = interpolated_shorelines.index.date

    plt.figure(figsize=(15, 5))  # Adjust the figure size as needed
    sns.heatmap(interpolated_shorelines.T, cmap='cool_r', cbar=True, vmin=-2.5, vmax=2.5)  # Transpose to get time on x-axis
    plt.title('NSW Shorelines')
    plt.ylabel('Transect')

    bounds = boundaries(settings)
    for thisedge in bounds:
        plt.axhline(y = thisedge , color= 'white', linestyle='-', linewidth=1.5)
    plt.show()

#####################################################
#####################################################


def plot_results(model_output, data, settings, unseen_target):

    plt.rcParams["font.family"] = "Times New Roman" 

    data_path = os.path.join(os.getcwd(),'data','model_input_data')
    static = pd.read_parquet(os.path.join(data_path,"Regional_NSW_dataset_static.parquet"), engine="pyarrow", use_threads=True)

    obs = data.df[unseen_target][data.first_valid_index:data.last_valid_index].interpolate(method='linear')
    obs = obs.loc[data.holdout.index[-settings['output_length']:]]
    
    residuals = abs(obs - model_output)
    # residuals = obs - model_output

    fig, axs = plt.subplots(3, figsize=(13, 14), facecolor=None, sharex=False)

    axs[0].set_title('NSW Coastline')
    axs[1].set_title('GXT-α')
    axs[2].set_title('Absolute Residual Error')

    for ax in axs:
        ax.grid(False)  # Disable grid for heatmaps only

    # ----> Heatmaps
    custom_cmap = LinearSegmentedColormap.from_list("custom_red_green", ["b",'b' ,"#c000ff"])
    obs_map = sns.heatmap(obs.T, cmap='winter', cbar=True, ax=axs[0], vmin=-32, vmax=32, zorder=0, cbar_kws={'label': 'Shoreline (m)'})
    pred_map = sns.heatmap(model_output.T, cmap='winter', cbar=True, ax=axs[1], vmin=-32, vmax=32, zorder=0, cbar_kws={'label': 'Shoreline (m)'})

    obs_map.collections[0].colorbar.ax.set_ylabel('Shoreline (m)', fontsize=10)
    obs_map.collections[0].colorbar.ax.tick_params(labelsize=10)

    pred_map.collections[0].colorbar.ax.set_ylabel('Shoreline (m)', fontsize=10)
    pred_map.collections[0].colorbar.ax.tick_params(labelsize=10)



    custom_cmap = LinearSegmentedColormap.from_list("custom_red_green", ["#000000", "b", 'white'])
    # custom_cmap = LinearSegmentedColormap.from_list("custom_red_green", ["blue", "#ae00ff", "red"])
    sns.heatmap(residuals.T, cmap=custom_cmap, cbar=True, ax=axs[2], vmin=10, vmax=32, zorder=0, cbar_kws={'label': 'Error (m)'})

    # ----> Y-Axis: Label Every 100th Transect
    # ytick_positions = np.arange(0, len(data.valid_samples), 300)
    # ytick_positions = np.arange(0, len(data.valid_samples), 300)  # Every 300th index
    # ytick_labels = np.array(data.valid_samples)[ytick_positions]  # Get corresponding labels

    ytick_positions = unseen_target[::300]  # Get every 300th index
    ytick_labels = static.loc[ytick_positions]['Latitude - Origin'].to_numpy()  # Get corresponding labels
    ytick_positions = np.arange(0, len(unseen_target), 300)
     
    for ax in axs:
        ax.set_yticks(ytick_positions)
        # ax.set_yticklabels(ytick_labels)
        ax.set_yticklabels([f"{label:.2f}" for label in ytick_labels], fontsize = 8)
        ax.set_ylabel('Latitude', fontsize = 10)

    # ----> X-Axis: Label Start of Every Year
    # ----> Compute yearly intervals starting from the first date in `obs.index`
    first_year_date = obs.index[0]  # First date in the index
    yearly_dates = pd.date_range(start=first_year_date, end=obs.index[-1], freq='YS')  # 'YS' = Year Start

    # Find the closest matching index positions in `obs.index`
    year_start_positions = [obs.index.get_indexer([date], method='nearest')[0] for date in yearly_dates]

    # Extract year labels
    year_labels = [date.strftime('%Y') for date in yearly_dates]

    # Apply to the last subplot (shared x-axis)
    axs[0].set_xticks(year_start_positions)
    axs[0].set_xticklabels(year_labels, rotation=45, fontsize = 10)
    axs[1].set_xticks(year_start_positions)
    axs[1].set_xticklabels(year_labels, rotation=45, fontsize = 10)
    axs[2].set_xticks(year_start_positions)
    axs[2].set_xticklabels(year_labels, rotation=45)
    # axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Ensure year format
    # axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Ensure year format
    
    
    plt.savefig('figures/heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# def plot_results(model_output, data, settings):



#     obs = data.calibration[data.valid_samples][data.first_valid_index:data.last_valid_index].interpolate(method='linear')
#     obs = obs.loc[data.test.index[-settings['output_length']:]]
#     # interpolated_shorelines.index = interpolated_shorelines.index.date
#     # obs = interpolated_shorelines.loc[model_output.index]

#     # model_output = destandardize(model_output, data.scalers, 'shoreline')
#     # obs = destandardize(obs, data.scalers, 'shoreline')
#     residuals = abs(obs-model_output)

#     fig, axs = plt.subplots(3,figsize=(9,10),facecolor=None, sharex = True)

#     axs[0].set_title('NSW Coastline')
#     axs[1].set_title('Model Output')
#     axs[2].set_title('Absolute Residual Error')

#     for ax in axs:
#         ax.grid(False)  # Disable grid for heatmaps only
#         ax.set_xticks([0, 1])


#     hax = sns.heatmap(obs.T, cmap='cool_r', cbar=True, ax=axs[0], vmin = -32, zorder = 0 ,vmax = 32, cbar_kws={'label': 'Chainage (m)'})
#     sns.heatmap(model_output.T, cmap='cool_r', cbar=True, vmin = -32, vmax = 32, zorder = 0, ax=axs[1], cbar_kws={'label': 'Chainage (m)'})


#     custom_cmap = LinearSegmentedColormap.from_list("custom_red_green", ["#000000", "#ae00ff", "red"])
#     sns.heatmap(residuals.T, cmap=custom_cmap, cbar=True, vmin = 10, vmax = 32, zorder = 0, ax=axs[2], cbar_kws={'label': 'Error (m)'})


#     # Select every 100th transect for labeling
#     ytick_positions = np.arange(0, len(data.valid_samples), 300)
#     print(ytick_positions) 
#     ytick_labels = obs.columns[ytick_positions]
    
#     for ax in axs:
#         ax.set_yticks(ytick_positions)  # Set tick positions
#         ax.set_yticklabels(ytick_labels)  # Assign corresponding labels
    

      # Assuming transect names are in columns
# Ensures no gridlines from Matplotlib
    # hax2 = hax.twinx()
    # hax2.axvline(x=obs.index[500], color='red', linestyle='--', linewidth=6, label='Test Start', zorder = 2)
    
    # axs[0].axvline(x=obs.index[500], color='red', linestyle='--', linewidth=6, label='Test Start', zorder = 2)
    
    
    # bounds = boundaries(data)
    # for ii, ax in enumerate(axs):
    #     for thisedge in bounds:
    #         ax.axhline(y = thisedge , color= ('white' if ii < 2 else 'k'), linestyle='-', linewidth=0.1)




#####################################################
#####################################################

def boundaries(data):

    edges = []
    for idx, item in enumerate(data.training_samples):
        if idx > 0:
            if item[5:9] != data.training_samples[idx-1][5:9]:
                edges.append(idx)
    
    return edges

#####################################################
#####################################################

# def inspect_variables(x, future, y, gt, settings):

#     n_inputs = x.shape[2]
#     x = torch.where(x == -9999, np.nan, x)
#     future = torch.where(future == -9999, np.nan, future)

#     prev_target = x[0,:,-1]
#     x = torch.concat((x[0,:,:-1],future[0]), dim = 0)
#     y = torch.where(y == -9999, np.nan, y)

#     fig, axs = plt.subplots(2, figsize = (5,3) , sharex = True)
#     plt.subplots_adjust(wspace=0.075, hspace=0.1)
#     plt.rcParams["font.family"] = "Times New Roman"
#     # plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
#     # tick_fontsize = 16
#     # axis_fontsize = 16

#     axs[0].scatter(range(0,520), gt[:520].values, color = 'k', facecolor = 'w', alpha = 0.5, s = 10, zorder = 1, marker = 's')

#     axs[0].plot(prev_target, lw = 1, c = '#CC0066')
#     axs[0].plot(range(260,520), y[0], lw = 1, c = 'k', linestyle = '-')

#     axs[1].plot(x[:,-2][:260], lw = 1, c = '#CC0066')
#     axs[1].plot(range(260,520), x[:,-2][260:], lw = 1, c = '#0b46a0')

#     axs[1].plot(x[:,-1][:260], lw = 1, c = '#CC0066')
#     axs[1].plot(range(260,520), x[:,-1][260:], lw = 1, c = '#0b46a0')

#     # axs[0].set_ylabel('Shoreline (m)', fontsize = 10)
#     axs[1].plot(x[:,0][:260]-7, lw = .75, c = '#CC0066')
#     axs[1].plot(x[:,1][:260]-14, lw = .75, c = '#CC0066')
#     axs[1].plot(x[:,2][:260]-21, lw = .75, c = '#CC0066')

#     axs[1].plot(range(260,520), x[:,0][260:]-7, lw = .75, c = '#0b46a0')
#     axs[1].plot(range(260,520), x[:,1][260:]-14, lw = .75, c = '#0b46a0')
#     axs[1].plot(range(260,520), x[:,2][260:]-21, lw = 0.75, c = '#0b46a0')
  
#     # for ii, _ in enumerate(range(n_inputs-3)):
#     #     idx = 2+ii
#     #     axs[idx].plot(x[:,ii][:260], lw = 0.5, c = '#CC0066') # color = np.random.rand(3,)
#     #     axs[idx].plot(range(260,520), x[:,ii][-260:], lw = 0.5, c = '#1159d6') # color = np.random.rand(3,)
        
#     #     # axs[idx].axvspan(0, 260, color='#CC0066', alpha=.05, lw=1, zorder = 0, fill=True)
#     #     # axs[idx].axvspan(260, 520, color='#1159d6', alpha=.05, lw=1, zorder = 0, fill=True)


#     #     axs[idx].set_ylabel(settings['dynamic_inputs'][ii])

        
#     for ax in axs:
#         ax.axvline(260, c='grey', linewidth = 1, zorder=5)
#         ax.set_facecolor('#ffffff')
#         ax.grid(linestyle = ':', linewidth = 1, axis='both', zorder = 0)
#         ax.axvspan(0, 260, color='grey', alpha=.1, lw=1, zorder = 0, fill=True , label = 'Calibration Data')
#         # ax.axvspan(0, 260, color='#CC0066', alpha=.025, lw=1, zorder = 0, fill=True, label = 'Historical Input')
#         # Major & minor ticks for X-axis
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
#         ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))
        
#         # Major & minor ticks for Y-axis (auto scale; adjust as needed)
#         ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # or use MultipleLocator if you want fixed intervals
#         ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

#         ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
        
#         # Style tick marks
#         ax.tick_params(axis='x', which='both', color='gray', direction = 'out')
#         ax.tick_params(axis='y', which='both', color='gray', direction = 'out')
#         ax.set_xlim(0,520)
    
#     # axs[0].axvspan(260, 520, color='grey', alpha=.1, lw=1, zorder = 0, fill=True, hatch = '\\\\', label = 'Target')
#     # axs[1].axvspan(260, 520, color='#1159d6', alpha=.05, lw=1, zorder = 0, fill=True)
    

#     axs[1].set_yticks([0, -7, -14, -21])  # Set the tick positions
#     axs[1].set_yticklabels(['$\\sin(t)$ & $\\cos(t)$', '$H_{s,mean}$', '$H_{s,peak}$', '$T_p$'])
#     axs[0].set_yticks([2,0,-2])  # Set the tick positions
#     axs[0].set_yticklabels(['30','0','-30'])
#     axs[0].set_ylim(-4,4)

#     plt.rcParams.update({
#         'axes.labelsize': 8,  # X and Y axis label size
#         'xtick.labelsize': 8,  # X-axis tick labels
#         'ytick.labelsize': 10,  # Y-axis tick labels
#         'legend.fontsize': 5   # Legend labels
#         })
    

#     fig.text(0.325, 0.0, 'Historical Inputs', fontstyle='italic', ha='center', fontsize=10)
#     fig.text(0.71, 0.0, 'Known Future Inputs', fontstyle='italic', ha='center', fontsize=10)
#     fig.text(0, 0.7, 'Shoreline (m)', fontstyle='italic', ha='center', fontsize=10)
#     fig.text(0.71, 0.9, 'Target', fontstyle='italic', ha='center', fontsize=10)
#     fig.text(0.325, 0.9, '$y_{prev}$', ha='center', fontsize=10)
#     # fig.text(0.5175, -0.05, 'Time', fontstyle='italic', ha='center', fontsize=10)

#     plt.savefig(f'Inputs.png', bbox_inches="tight", dpi = 300)

#     # axs[0].legend()

#####################################################
#####################################################



def generate_fixed_val_spatial_folds(static_df, training_sites, n_folds=5, n_val_sites=40, n_clusters=60, random_state=42):
    # Filter to only training sites
    filtered_df = static_df[static_df.index.str[:7].isin(training_sites)].copy()

    # Group by beach (first 7 chars of transect ID)
    beach_static = filtered_df.groupby(filtered_df.index.str[:7]).mean()

    # Normalize features for clustering
    features = (beach_static - beach_static.mean()) / beach_static.std()

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    beach_static['cluster'] = kmeans.fit_predict(features)

    folds = []

    for fold_idx in range(n_folds):
        np.random.seed(random_state + fold_idx)

        # Shuffle clusters
        cluster_ids = beach_static['cluster'].unique()
        np.random.shuffle(cluster_ids)

        val_sites = []
        for cluster_id in cluster_ids:
            candidates = beach_static[beach_static['cluster'] == cluster_id].index.tolist()
            np.random.shuffle(candidates)
            for site in candidates:
                if site not in val_sites:
                    val_sites.append(site)
                    break
            if len(val_sites) >= n_val_sites:
                break

        train_sites = [site for site in beach_static.index if site not in val_sites]
        folds.append([sorted(train_sites), sorted(val_sites)])  # exactly what you wanted

    return folds