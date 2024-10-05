from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from collections import Counter


import pandas as pd
import numpy as np
import geopandas as gpd
import scipy



###  recall training dataframe

data = pd.read_pickle(f"/content/drive/MyDrive/training_data_grids_final.pkl")
data_lags_nn_queen_less_rook = pd.read_pickle(f"/content/drive/MyDrive/training_data_grids_nn_queen_less_rook.pkl")
data_lags_nn_rook = pd.read_pickle(f"/content/drive/MyDrive/training_data_grids_nn_rook.pkl")

data = data.drop([x for x in data.columns if 'neighb' in x or 'edge' in x or 'geom' in x], axis=1)
data = pd.concat([data[[c for c in data.columns if "nn" not in c]], data_lags_nn_queen_less_rook, data_lags_nn_rook], axis=1)

data['day'] = data['time'].apply(lambda x: x.day)
data['day_of_year'] = data['time'].apply(lambda x: x.timetuple().tm_yday)
data['month'] = data['time'].apply(lambda x: x.month)
data['week'] = data['time'].apply(lambda x: x.isocalendar()[1])
data['weekday'] = data['time'].apply(lambda x: x.weekday())

data = data.merge(weighted_so2, on=['id','time'], how='left')

data['SO2_above_0'] = data['mean_so2'].apply(lambda x: 0 if x == 0 else 1)




##  binary model

def cross_validate_fit_binary_classification(model, X_train, y_train):

    accuracy = []
    precision = []
    recall = []
    f1score = []

    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_cv, y_train_cv = X_train[train_index], y_train.values[train_index]
        X_test_cv, y_test_cv = X_train[test_index], y_train.values[test_index]

        model.fit(X_train, y_train)

        binary_preds_train = model.predict(X_train_cv)
        binary_preds_test = model.predict(X_test_cv)

        m1 = (accuracy_score(y_test_cv, binary_preds_test), accuracy_score(y_train_cv, binary_preds_train))
        m2 = (precision_score(y_test_cv, binary_preds_test, average="weighted"), precision_score(y_train_cv, binary_preds_train, average="weighted"))
        m3 = (recall_score(y_test_cv, binary_preds_test, average="weighted"), recall_score(y_train_cv, binary_preds_train, average="weighted"))
        m4 = (f1_score(y_test_cv, binary_preds_test, average="weighted"), f1_score(y_train_cv, binary_preds_train, average="weighted"))

        accuracy.append(m1)
        precision.append(m2)
        recall.append(m3)
        f1score.append(m4)

    return accuracy, precision, recall, f1score



y = data['SO2_above_0']
X = data.iloc[:,6:323]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



lasso_logistic = Pipeline([("imputer", KNNImputer(n_neighbors=4, weights='uniform')),
                           ("regressor", LogisticRegression(penalty='l1', solver='liblinear'))])



gbmc = HistGradientBoostingClassifier(max_iter = 5000, learning_rate = 0.001,
                                      max_depth = 5, max_leaf_nodes = 2 ** 5 - 1,
                                      random_state=42)



##  regression model

def cross_validate_fit_regression(model, X_train, y_train):

    m1, m2, m3, m4, m5, m6, m7 = [], [], [], [], [], [], []

    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_cv, y_train_cv = X_train[train_index], y_train.values[train_index]
        X_test_cv, y_test_cv = X_train[test_index], y_train.values[test_index]

        model.fit(X_train, y_train)

        gbmlogmean_preds_train = model.predict(X_train_cv)
        gbmlogmean_preds_test = model.predict(X_test_cv)

        print("Train R2: " + str(r2_score(y_train_cv, gbmlogmean_preds_train)))
        print("Train R: " + str(scipy.stats.pearsonr(np.array(y_train_cv).flatten(), gbmlogmean_preds_train)[0]))
        print("Train MAE: " + str(mean_absolute_error(y_train_cv, gbmlogmean_preds_train)))
        print("Train MedAE: " + str(median_absolute_error(y_train_cv, gbmlogmean_preds_train)))
        print("Train MAE (original): " + str(mean_absolute_error(np.exp(y_train_cv) - 0.001, np.exp(gbmlogmean_preds_train) - 0.001)))
        print("Train MedAE (original): " + str(median_absolute_error(np.exp(y_train_cv) - 0.001, np.exp(gbmlogmean_preds_train) - 0.001)))
        print("Train upper quartile (original): " + str(np.quantile(abs(np.exp(y_train_cv) - np.exp(gbmlogmean_preds_train)), 0.75)))
        print("\n")
        print("Test R2: " + str(r2_score(y_test_cv, gbmlogmean_preds_test)))
        print("Test R: " + str(scipy.stats.pearsonr(np.array(y_test_cv).flatten(), gbmlogmean_preds_test)[0]))
        print("Test MAE: " + str(mean_absolute_error(y_test_cv, gbmlogmean_preds_test)))
        print("Test MedAE: " + str(median_absolute_error(y_test_cv, gbmlogmean_preds_test)))
        print("Test MAE (original): " + str(mean_absolute_error(np.exp(y_test_cv) - 0.001, np.exp(gbmlogmean_preds_test) - 0.001)))
        print("Test MedAE (original): " + str(median_absolute_error(np.exp(y_test_cv) - 0.001, np.exp(gbmlogmean_preds_test) - 0.001)))
        print("Test upper quartile (original): " + str(np.quantile(abs(np.exp(y_test_cv) - np.exp(gbmlogmean_preds_test)), 0.75)))
        print("\n")

        m1.append((r2_score(y_test_cv, gbmlogmean_preds_test), 
                   r2_score(y_train_cv, gbmlogmean_preds_train)))
        m2.append((scipy.stats.pearsonr(np.array(y_test_cv).flatten(), 
                                        gbmlogmean_preds_test)[0], scipy.stats.pearsonr(np.array(y_train_cv).flatten(), gbmlogmean_preds_train)[0]))
        m3.append((mean_absolute_error(y_test_cv, gbmlogmean_preds_test), 
                   mean_absolute_error(y_train_cv, gbmlogmean_preds_train)))
        m4.append((median_absolute_error(y_test_cv, gbmlogmean_preds_test), 
                   median_absolute_error(y_train_cv, gbmlogmean_preds_train)))
        m5.append((mean_absolute_error(np.exp(y_test_cv) - 0.001, np.exp(gbmlogmean_preds_test) - 0.001), 
                   mean_absolute_error(np.exp(y_train_cv) - 0.001, np.exp(gbmlogmean_preds_train) - 0.001)))
        m6.append((median_absolute_error(np.exp(y_test_cv) - 0.001, np.exp(gbmlogmean_preds_test) - 0.001),
                   median_absolute_error(np.exp(y_train_cv) - 0.001, np.exp(gbmlogmean_preds_train) - 0.001)))
        m7.append((np.quantile(abs(np.exp(y_test_cv) - np.exp(gbmlogmean_preds_test)), 0.75),
                   np.quantile(abs(np.exp(y_train_cv) - np.exp(gbmlogmean_preds_train)), 0.75)))

    return m1, m2, m3, m4, m5, m6, m7



endog = np.log(data[data['SO2_above_0'] == 1]['mean_so2'] + 0.001)
scaler = StandardScaler().fit(data[data['SO2_above_0'] == 1].iloc[:,6:318])
exog = np.concatenate((
        scaler.transform(data[data['SO2_above_0'] == 1].iloc[:,6:318]),
        data[data['SO2_above_0'] == 1].iloc[:,318:359],
), axis = 1)
group = data[data['SO2_above_0'] == 1]['id']
X_train, X_test, y_train, y_test = train_test_split(exog, endog, test_size=0.2, random_state=42, stratify=group)



endog = np.log(data[data['SO2_above_0'] == 1]['max_so2'] + 0.001)
scaler = StandardScaler().fit(data[data['SO2_above_0'] == 1].iloc[:,6:318])
exog = np.concatenate((
        scaler.transform(data[data['SO2_above_0'] == 1].iloc[:,6:318]),
        data[data['SO2_above_0'] == 1].iloc[:,318:359],
), axis = 1)
group = data[data['SO2_above_0'] == 1]['id']
X_train, X_test, y_train, y_test = train_test_split(exog, endog, test_size=0.2, random_state=42, stratify=group)



lasso = Pipeline([("imputer", KNNImputer(n_neighbors=8, weights='uniform')),
                  ("regressor", Lasso(alpha=0.1, max_iter=6000, tol=0.00001))])



gbm = HistGradientBoostingRegressor(max_iter = 5000, learning_rate = 0.001,
                                    max_depth = 5, max_leaf_nodes = 2 ** 5 - 1,
                                    random_state=42)

