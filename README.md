### A two-stage Gradient Boosting model for Sulphur Dioxide hot spot mapping in Saint John, New Brunswick

---------------------------

This repository archives the codes for implementing the $SO_{2}$ hot spot mapping based in Saint John, New Brunswick, an industrial city in Canada.

To generate the hot spots, we developed a model for estimating the surface $SO_{2}$ concentrations employing the histrogram-based Gradient Boosting Model (HGBM) in the scikit-learn package in Python environment.
The model comprised of two stages, a binary classifier for determining $SO_{2}$-clean and $SO_{2}$-polluted scenarios, then a regressor on aggregated mean and maximum per monitoring stations in the training dataset.
We also provide implementation method for hot spot visualization.

This is part of the submission of a manuscript. A pre-printed version is available here:

> <i>TBC</i>
<br><br>


Please refer to the following hierarchy of the repository structure:

    .
    ├── access                   
        ├──  download.py            # module for downloading data from open-source datasets
        
    ├── augmentation                   
        ├──  model.py               # module for data augmentation
        ├──  demo.ipynb             # result documentation
        
    ├── training
        ├──  model.py               # module for model training
        ├──  preprocess.py          # module for preprocessing
        ├──  predict.py             # module for generating predictions
        ├──  demo.ipynb             # result documentation

    ├── hotspot
        ├──  hotspot.py             # module for generating and visualizing hot spots
        ├──  demo.ipynb             # result documentation
        
    ├── src                         # data sources (accessed from non-automated process)




