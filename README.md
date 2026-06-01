# A Generalized Data-Driven Shoreline Model at the Regional Scale

**Kit Calcraft<sup>1</sup>, Joshua A. Simmons<sup>2</sup>, Lucy A. Marshall<sup>3</sup>, Kristen D. Splinter<sup>1</sup>**  
<sup>1</sup>Water Research Laboratory, School of Civil and Environmental Engineering, UNSW Sydney  
<sup>2</sup>ARC Training Centre in Data Analytics for Resources and Environments (DARE), Sydney, NSW, Australia  
<sup>3</sup>Faculty of Engineering, The University of Sydney  

---

## Description

This repository accompanies the manuscript *“A Generalised Data-Driven Shoreline Model at the Regional Scale”*.  
It provides code, workflows, and interactive outputs associated with the first data-driven shoreline model trained across ~2,000 km of the southeast Australian coastline.  

The model adapts the **Temporal Fusion Transformer (TFT)** architecture, integrating dynamic wave forcing with static site descriptors to produce forecasts that generalise across diverse coastal environments **without site-specific calibration**.  
Evaluation on unseen sites demonstrates reliable transferability at a regional scale, while ablation experiments highlight the importance of both dynamic wave inputs and static covariates in supporting generalisation.  

---

## Features

- End-to-end workflow for model training, weight loading, and evaluation.  
- Ablation experiments to isolate the contribution of different input groups.  
- Interactive map of **over 300 beaches** along the NSW coast, with click-enabled forecasts.  
- Visualisation of predicted shoreline positions, historical observations, and uncertainty bands.  
- Built with:
  - PyTorch (model training/inference)  
  - Folium + Chart.js (map and timeseries outputs)  
  - JSON-based data delivery  

---

## How to Use

1. **Explore Interactive Outputs**  
   Visit the GitHub Pages site: [https://kitnotpat.github.io/A_Generalised_ShorelineModel/](https://kitnotpat.github.io/A_Generalised_ShorelineModel/)  
   - Zoom and pan along the coastline.  
   - Click on a beach to view forecasts, ground-truth shoreline observations, and prediction intervals.  

2. **Reproduce Model Training**  
   - Settings and hyperparameters are stored in `config/model_settings.yml`.  
   - Main workflow is provided in the notebook `shoreline_model_pipeline.ipynb`, which covers:
     - Configuration loading  
     - Data preparation  
     - Model build (TFT variant)  
     - Training or pretrained weight loading  
     - Evaluation and visualisation  

---

## Data  

The model requires a large dataframe containing **all transects** and associated dynamic forcing variables.  
Each row corresponds to a weekly timestep, and columns include:  

(Yes I know I should've used a xarray, I was impatient and got to deep, lesson learned)

- `date`, `sin_t`, `cos_t` (time encodings)  
- Shoreline position per transect (e.g. `aus0052-0088`)  
- Associated wave parameters per transect (e.g. `aus0052-0088_Dp_mean`, `aus0052-0088_Hs_mean`, `aus0052-0088_Hs_peak`, `aus0052-0088_Tp_mean`)  

In practice, the full dataset spans **thousands of transects** and all relevant dynamic variables.  
Due to size, this data is archived on Zenodo and can be downloaded here:  
👉 [https://zenodo.org/records/16749486](https://zenodo.org/records/16749486)  

To illustrate the required structure, we provide an **example transect dataset** (`aus0052-0088`) in this repository.  

| date       | sin_t   | cos_t   | aus0052-0088 | aus0052-0088_Dp_mean | aus0052-0088_Hs_mean | aus0052-0088_Hs_peak | aus0052-0088_Tp_mean |
|------------|---------|---------|---------------|-----------------------|-----------------------|-----------------------|-----------------------|
| 1987-09-11 | -0.94   | -0.34   | 30.06         | 84.93                 | 0.45                  | 0.49                  | 11.10                 |
| 1987-09-18 | -0.98   | -0.22   | NaN           | 67.42                 | 0.48                  | 1.04                  | 7.64                  |
| 1987-09-25 | -0.99   | -0.10   | NaN           | 67.98                 | 0.52                  | 0.74                  | 7.36                  |
| 1987-10-02 | -1.00   |  0.02   | 41.90         | 66.38                 | 0.65                  | 1.26                  | 6.84                  |

This toy dataset allows users to run the workflow end-to-end on a single transect.  
For **full training or evaluation**, replace it with the complete dataframe downloaded from Zenodo.

## Citation

If you use this model, code, or outputs, please cite:  

Calcraft, K., Simmons, J. A., Marshall, L. A., & Splinter, K. D. (2026). A generalized data-driven shoreline model at the regional scale. Journal of Geophysical Research: Machine Learning and Computation, 3, e2025JH000989. [https://doi.org/10.1029/2025JH000989]
