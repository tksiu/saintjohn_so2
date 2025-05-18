### A two-stage Gradient Boosting model for Sulphur Dioxide hot spot mapping in Saint John, New Brunswick, Canada

---------------------------

This repository archives the codes for implementing the $SO_{2}$ hot spot mapping based in Saint John, New Brunswick, an industrial city in Canada.

To generate the hot spots, we developed a model for estimating the surface $SO_{2}$ concentrations employing the histrogram-based Gradient Boosting Model (HGBM) in the scikit-learn package in Python environment.
The model comprised of two stages, a binary classifier for determining $SO_{2}$-clean and $SO_{2}$-polluted scenarios, then a regressor on aggregated mean and maximum per monitoring stations in the training dataset.
We also provide implementation method for hot spot visualization.

The published manuscript is available at:

> <i>Siu, T.K., Greene, C.S., Fong, K.C., 2025. Identifying surface sulphur dioxide (SO2) monitoring gaps in Saint John, Canada with land use regression and hot spot mapping. Atmospheric Environment 353, 121238. <a her="https://doi.org/10.1016/j.atmosenv.2025.121238">https://doi.org/10.1016/j.atmosenv.2025.121238</a></i>
<br><br>


Please refer to the following hierarchy of the repository structure:

    .
    ├── access                   
        ├──  gee_export.py            # module for downloading data from Google Earth Engine datasets
        ├──  osm_export.py            # module for downloading data from OpenStreetMap datasets
        ├──  reference_grid.py        # module for generating a grid scale as output resolution
        ├──  docuentation.ipynb       # result documentation
        
    ├── augmentation                   
        ├──  model.py                   # module for data augmentation
        ├──  preprocess.py              # module for generating training dataframe
        ├──  docuentation.ipynb         # result documentation
        
    ├── training
        ├──  model.py                                     # module for model training
        ├──  zonal_stats.py                               # module for preprocessing raster/vector features
        ├──  prediction.py                                # module for generating predictions
        ├──  documentation_get_features.ipynb             # consolidating features into dataframe (inputs)
        ├──  documentation_get_outcome.ipynb              # consolidating outcome (monitored & augmented)
        ├──  documentation_model.ipynb                    # training performance & results
        ├──  documentation_zonal_stats.ipynb              # demonstrating zonal statistics
        ├──  documentation_feature_importance_table.xlsx  # logging feature importance results

    ├── hotspot
        ├──  hotspot.py                     # module for generating and visualizing hot spots
        ├──  docuentation.ipynb             # result documentation

    ├── wind                        # supplementary analysis for wind-drive northeastward transport
        
    ├── src                         # data sources (accessed from non-automated process)
    ├── archive                     # archived pre-trained models


> Please use <b>sklearn 1.3.0</b> for loading the pre-trained <b>augmentation</b> models. <br>
> Please use <b>sklearn 1.5.0</b> for loading the pre-trained <b>surface SO2 (daily mean/max)</b> models.



