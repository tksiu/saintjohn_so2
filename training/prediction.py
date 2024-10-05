import numpy as np
import pandas as pd
import pickle



with open("./archive/gbmclassify.pkl", 'rb') as f:
    gbm_classifier = pickle.load(f)

with open("./archive/gbmregressor_mean_so2.pkl", 'rb') as f:
    gbm_regressor_mean_so2 = pickle.load(f)

with open("./archive/gbmregressor_max_so2.pkl", 'rb') as f:
    gbm_regressor_max_so2 = pickle.load(f)

with open("./archive/gbmregressor_mean_so2_scaler.pkl", 'rb') as f:
    scaler_gbmregressor_mean_so2 = pickle.load(f)
    
with open("./archive/gbmregressor_max_so2_scaler.pkl", 'rb') as f:
    scaler_gbmregressor_max_so2 = pickle.load(f)



preds_above_zero = []
preds_mean_so2 = []
preds_max_so2 = []

for chunk in pd.read_csv(f"/content/drive/MyDrive/complete_grid_data.csv", chunksize=1e5):
    chunk = chunk.drop(['Unnamed: 0'], axis=1)

    c_out = gbm_classifier.predict(chunk.iloc[:, 4:])
    preds_above_zero.append(c_out)

    x_concat = np.concatenate((
        scaler_gbmregressor_mean_so2.transform(chunk.iloc[:, 4:316]),
        chunk.iloc[:, 316:]
    ), axis = 1)

    mean_out = gbm_regressor_mean_so2.predict(x_concat)
    max_out = gbm_regressor_max_so2.predict(x_concat)

    preds_mean_so2.append(mean_out)
    preds_max_so2.append(max_out)



df_preds = pd.DataFrame({"so2_above_0": preds_above_zero,
                        "mean_so2": preds_mean_so2,
                        "max_so2": preds_max_so2})

df_preds["mean_so2"] = df_preds["mean_so2"].apply(lambda x: np.exp(x) - 0.001)
df_preds["max_so2"] = df_preds["max_so2"].apply(lambda x: np.exp(x) - 0.001)

