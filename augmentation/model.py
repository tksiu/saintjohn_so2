from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import numpy as np
from collections import Counter


from preprocess import ts_data


###  Binary Classifier

endog = ts_data['SO2_above_0']
exog = np.log(ts_data[[x for x in ts_data.columns if "TRS" in x or "weighted_SO2" in x]] + 0.0001)
exog = np.concatenate((
        np.log(ts_data[[x for x in ts_data.columns if x == "TRS" or ("TRS" in x and "1hr" in x)]] + 0.0001), 
        np.log(ts_data[[x for x in ts_data.columns if x == "weighted_SO2" or ("weighted_SO2" in x and "1hr" in x)]] + 0.0001), 
        np.log(ts_data[[x for x in ts_data.columns if "TRS_" in x and "1hr" not in x and "additive" not in x and "multiplicative" not in x]] + 0.0001), 
        np.log(ts_data[[x for x in ts_data.columns if "weighted_SO2_" in x and "1hr" not in x and "additive" not in x and "multiplicative" not in x]] + 0.0001), 
        np.array(ts_data[[x for x in ts_data.columns if "TRS" in x and ("additive" in x or "multiplicative" in x)]]),
        np.array(ts_data[[x for x in ts_data.columns if "weighted_SO2" in x and ("additive" in x or "multiplicative" in x)]]),
        np.array(ts_data[['month','week','weekday','day_of_year','hour']])
    ), axis = 1)

group = ts_data['station'].astype(str) + " - " + ts_data['SO2_above_0'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(exog, endog, test_size=0.2, random_state=42, stratify=group)


gbmc = HistGradientBoostingClassifier(
    max_iter=50000, 
    learning_rate=0.001, 
    max_depth=10, 
    random_state=42, 
    class_weight={0: 1 / (Counter(ts_data['SO2_above_0'])[0] / (Counter(ts_data['SO2_above_0'])[1] + Counter(ts_data['SO2_above_0'])[0])), 
                  1: 1 / (Counter(ts_data['SO2_above_0'])[1] / (Counter(ts_data['SO2_above_0'])[1] + Counter(ts_data['SO2_above_0'])[0]))}
)
gbmc.fit(X_train, y_train)
gbmc_preds_train = gbmc.predict(X_train)
gbmc_preds_test = gbmc.predict(X_test)


###  Regressor

endog = np.log(ts_data[ts_data['SO2_above_0'] == 1]['SO2'] + 0.001)
exog = np.log(ts_data[ts_data['SO2_above_0'] == 1][[x for x in ts_data.columns if "TRS" in x or "weighted_SO2" in x]] + 0.0001)
exog = np.concatenate((
        np.log(ts_data[ts_data['SO2_above_0'] == 1][[x for x in ts_data.columns if x == "TRS" or ("TRS" in x and "1hr" in x)]] + 0.0001), 
        np.log(ts_data[ts_data['SO2_above_0'] == 1][[x for x in ts_data.columns if x == "weighted_SO2" or ("weighted_SO2" in x and "1hr" in x)]] + 0.0001), 
        np.log(ts_data[ts_data['SO2_above_0'] == 1][[x for x in ts_data.columns if "TRS_" in x and "1hr" not in x and "additive" not in x and "multiplicative" not in x]] + 0.0001), 
        np.log(ts_data[ts_data['SO2_above_0'] == 1][[x for x in ts_data.columns if "weighted_SO2_" in x and "1hr" not in x and "additive" not in x and "multiplicative" not in x]] + 0.0001), 
        np.array(ts_data[ts_data['SO2_above_0'] == 1][[x for x in ts_data.columns if "TRS" in x and ("additive" in x or "multiplicative" in x)]]),
        np.array(ts_data[ts_data['SO2_above_0'] == 1][[x for x in ts_data.columns if "weighted_SO2" in x and ("additive" in x or "multiplicative" in x)]]),
        np.array(ts_data[ts_data['SO2_above_0'] == 1][['month','week','weekday','day_of_year','hour']])
    ), axis = 1)

group = ts_data[ts_data['SO2_above_0'] == 1]['station']

X_train, X_test, y_train, y_test = train_test_split(exog, endog, test_size=0.2, random_state=42, stratify=group)

gbm = HistGradientBoostingRegressor(
    max_iter=30000, 
    learning_rate=0.001, 
    max_depth=10, 
    random_state=42
)
gbm.fit(X_train, y_train)
gbm_preds_train = gbm.predict(X_train)
gbm_preds_test = gbm.predict(X_test)

