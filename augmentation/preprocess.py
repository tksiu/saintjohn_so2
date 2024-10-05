import os
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import math
import scipy

import statsmodels.api as sm
import statsmodels.gam.api as smgam
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests

from functools import reduce



def get_training_data():

    nb_stations = pd.read_excel(f"./src/NB_Air_Quality_stations_metadata.xlsx")

    nb_records_path = f"./src/SJ FEM Station Data/"
    nb_so2_files = [x for x in os.listdir(nb_records_path) if "NB FEM SO2" in x]
    nb_trs_files = [x for x in os.listdir(nb_records_path) if "NB FEM TRS" in x]

    nb_so2_records = []
    nb_so2_stations = []

    for s in range(len(nb_so2_files)):
        nb_station_records = pd.read_excel(nb_records_path + nb_so2_files[s])
        nb_station_records = nb_station_records.iloc[0:-1,:]
        
        nb_map_stations = list(nb_station_records.columns)[1:]
        nb_map_stations = [x for x in nb_map_stations if x != "Colson Cove - Musquash"]
        nb_map_stations = [x.replace("–", "-").replace(" - IOL", "") for x in nb_map_stations]
        
        nb_stations_tem = nb_stations[(nb_stations['city'] + " - " + nb_stations['station_name']).isin(nb_map_stations) & 
                                        (nb_stations['measurements'] == "SO2")]
        nb_stations_tem = nb_stations_tem[~((nb_stations_tem['station_name'] == "Forest Hills") & 
                                            (nb_stations_tem['owner'] == "Government of New Brunswick"))]

        nb_stations_tem['geometry'] = list(zip(nb_stations_tem['longitude'], nb_stations_tem['latitude']))
        nb_stations_tem['geometry'] = nb_stations_tem['geometry'].apply(shapely.Point)
        nb_stations_tem = gpd.GeoDataFrame(nb_stations_tem, geometry='geometry', crs = 'epsg:4326')
        
        nb_so2_records.append(nb_station_records)
        nb_so2_stations.append(nb_stations_tem)
        
    nb_so2_records = pd.concat(nb_so2_records)
    nb_so2_stations = pd.concat(nb_so2_stations).drop_duplicates()
    nb_so2_records.columns = ["Result_Date"] + nb_map_stations

    nb_TRS_records = pd.read_excel(nb_records_path + nb_trs_files[0])
    nb_TRS_records = nb_TRS_records.iloc[0:-1,:]
    nb_TRS_map_stations = list(nb_TRS_records.columns)[1:]
    nb_TRS_map_stations = [x.replace("–", "-").replace(" - IOL", "") for x in nb_TRS_map_stations]

    nb_TRS_stations = nb_stations[(nb_stations['city'] + " - " + nb_stations['station_name']).isin(nb_TRS_map_stations)]
    nb_TRS_stations = nb_TRS_stations.drop(["measurements"], axis=1).drop_duplicates()
    nb_TRS_stations = nb_TRS_stations[~((nb_TRS_stations['station_name'] == "Forest Hills") & 
                                        (nb_TRS_stations['owner'] == "Government of New Brunswick"))]

    nb_TRS_stations['geometry'] = list(zip(nb_TRS_stations['longitude'], nb_stations_tem['latitude']))
    nb_TRS_stations['geometry'] = nb_TRS_stations['geometry'].apply(shapely.Point)
    nb_TRS_stations = gpd.GeoDataFrame(nb_TRS_stations, geometry='geometry', crs = 'epsg:4326')

    nb_TRS_records.columns = ["Result_Date"] + nb_TRS_map_stations

    ts_train_data_so2 = nb_so2_records
    ts_train_data_trs = nb_TRS_records[[x for x in nb_TRS_records.columns if x in nb_so2_records.columns]]
    ts_train_data_so2['pollutant'] = "SO2"
    ts_train_data_trs['pollutant'] = "TRS"

    ts_train_data = pd.concat([ts_train_data_so2, ts_train_data_trs])

    train_stations = [x for x in nb_so2_records.columns if x in nb_TRS_records.columns and x != "Result_Date"]
    train_stations = nb_stations[nb_stations['station_name'].isin(train_stations)][['station_name','latitude','longitude']].drop_duplicates()
    train_stations = train_stations.iloc[1:,:]

    all_stations = nb_stations[nb_stations['station_name'].isin(ts_train_data['station'])][['station_name','latitude','longitude']].drop_duplicates()
    all_stations = all_stations.iloc[1:,:]

    return ts_train_data, \
            train_stations,  \
                all_stations



def dist(x, y):

    lat1 = math.radians(x[0])
    lon1 = math.radians(x[1])
    lat2 = math.radians(y[0])
    lon2 = math.radians(y[1])
    R = 6373.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return round(distance, 4)



def get_station_distance_matrix(all_stations):

    points = all_stations['station_name']
    distances = scipy.spatial.distance.pdist(all_stations[['latitude','longitude']].values, metric=dist)
    station_distances = pd.DataFrame(scipy.spatial.distance.squareform(distances), columns=points, index=points)

    return station_distances



def process_training_features(ts_data, start_time: str, end_time: str):
    
    ts_data = ts_data[(ts_data["Result_Date"] >= start_time) &
                      (ts_data["Result_Date"] <= end_time)]
    
    ## transformation
    
    ts_data = pd.melt(ts_data, id_vars=["Result_Date","pollutant"])
    ts_data = pd.pivot(ts_data, index=["variable","Result_Date"], columns=["pollutant"]).reset_index()
    ts_data.columns = ['station', 'date_hour'] + [c[1] for c in ts_data.columns][2:]

    ## creation of new variables
    
    ts_data['year_month'] = ts_data['date_hour'].apply(lambda x: str(x.year) + "-" + str(x.month))
    ts_data['year_week'] = ts_data['date_hour'].apply(lambda x: str(x.year) + "-" + str(x.week))
    ts_data['day'] = ts_data['date_hour'].apply(lambda x: x.date())
    ts_data['day_of_year'] = ts_data['day'].apply(lambda x: x.timetuple().tm_yday)
    ts_data['month'] = ts_data['date_hour'].apply(lambda x: x.month)
    ts_data['hour'] = ts_data['date_hour'].apply(lambda x: x.hour)
    ts_data['week'] = ts_data['date_hour'].apply(lambda x: x.week)
    ts_data['weekday'] = ts_data['date_hour'].apply(lambda x: x.weekday())

    ## periodic aggregation

    ts_data_local_daily_mean = ts_data.groupby(["station","day"]).agg({"SO2": np.nanmean, "TRS": np.nanmean})
    ts_data_local_daily_mean = ts_data_local_daily_mean.reset_index()

    ts_data_local_weekly_mean = ts_data.groupby(["station","year_week"]).agg({"SO2": np.nanmean, "TRS": np.nanmean})
    ts_data_local_weekly_mean = ts_data_local_weekly_mean.reset_index()

    ts_data_local_monthly_mean = ts_data.groupby(["station","year_month"]).agg({"SO2": np.nanmean, "TRS": np.nanmean})
    ts_data_local_monthly_mean = ts_data_local_monthly_mean.reset_index()

    ## fill missing value progressively

    ts_data['SO2'] = ts_data.apply(
        lambda x: ts_data_local_daily_mean[((ts_data_local_daily_mean['station'] == x['station']) & 
                                            (ts_data_local_daily_mean['day'] == x['day']))]['SO2'].values[0]
                            if pd.isnull(x['SO2']) else x['SO2'], 
                            axis = 1)
    ts_data['TRS'] = ts_data.apply(
        lambda x: ts_data_local_daily_mean[((ts_data_local_daily_mean['station'] == x['station']) & 
                                            (ts_data_local_daily_mean['day'] == x['day']))]['TRS'].values[0]
                            if pd.isnull(x['TRS']) else x['TRS'], 
                            axis = 1)

    ts_data['SO2'] = ts_data.apply(
        lambda x: ts_data_local_weekly_mean[((ts_data_local_weekly_mean['station'] == x['station']) & 
                                             (ts_data_local_weekly_mean['year_week'] == x['year_week']))]['SO2'].values[0]
                            if pd.isnull(x['SO2']) else x['SO2'], 
                            axis = 1)
    ts_data['TRS'] = ts_data.apply(
        lambda x: ts_data_local_weekly_mean[((ts_data_local_weekly_mean['station'] == x['station']) & 
                                             (ts_data_local_weekly_mean['year_week'] == x['year_week']))]['TRS'].values[0]
                            if pd.isnull(x['TRS']) else x['TRS'], 
                            axis = 1)

    ts_data['SO2'] = ts_data.apply(
        lambda x: ts_data_local_monthly_mean[((ts_data_local_monthly_mean['station'] == x['station']) & 
                                              (ts_data_local_monthly_mean['year_month'] == x['year_month']))]['SO2'].values[0]
                        if pd.isnull(x['SO2']) else x['SO2'], 
                        axis = 1)
    ts_data['TRS'] = ts_data.apply(
        lambda x: ts_data_local_monthly_mean[((ts_data_local_monthly_mean['station'] == x['station']) & 
                                              (ts_data_local_monthly_mean['year_month'] == x['year_month']))]['TRS'].values[0]
                        if pd.isnull(x['TRS']) else x['TRS'], 
                        axis = 1)
    
    return ts_data



def inverse_distance_weighted_so2(train_stations, station_distances, ts_data):

    weighted_so2 = []

    for s in train_stations['station_name']:

        ### for each station, calculated inverse distance weighting from other stations within 4 km buffer

        station_ts = ts_data[ts_data['station'] == s]
        station_inclusion = station_distances.loc[s, :][station_distances.loc[s, :] < 4].index.tolist()
        station_weightings = station_distances.loc[s, [x for x in station_distances.columns if x != s and x in station_inclusion]]
        station_weightings = 1 / (station_weightings / station_weightings.sum())
        station_weightings = station_weightings / station_weightings.sum()
        nearby_counter = 1

        ### for each station, calculated inverse distance weighted SO2 from the buffering stations

        for w in [x for x in station_inclusion if x != s]:
            station_ts_w = ts_data[ts_data['station'] == w][['date_hour','SO2']]
            station_ts_w['SO2'] = station_ts_w['SO2'] * station_weightings.loc[w]
            station_ts_w.columns = ['date_hour','nearby_station_' + str(nearby_counter) + "_SO2"]
            station_ts = station_ts.merge(station_ts_w, on=['date_hour'], how="left")
            nearby_counter += 1

        lng = len([x for x in station_inclusion if x != s])
        station_ts['weighted_SO2'] = station_ts.iloc[:, (station_ts.shape[1]-lng):station_ts.shape[1]].mean(axis=1)
        weighted_so2.append(station_ts)

    weighted_so2 = pd.concat(weighted_so2)
    weighted_so2 = weighted_so2[['station','date_hour','weighted_SO2']]

    ts_data = ts_data[ts_data['station'].isin(train_stations['station_name'])]
    ts_data = ts_data.merge(weighted_so2, on=['station','date_hour'], how='left')

    return ts_data



def lag_feature_engineering(ts_data, 
                            n_lags: int, 
                            rolling_windows = [4, 8, 12, 24],
                            rolling_aggregates = ["mean", "max"]):
    
    ###  calculate temporal (autocorrelated) lags

    for lag in range(n_lags):
        ts_data["TRS_1hr_lag_" + str(lag+1)] = ts_data.groupby(["station"])["TRS"].shift(lag)
        ts_data["weighted_SO2_1hr_lag_" + str(lag+1)] = ts_data.groupby(["station"])["weighted_SO2"].shift(lag)
    
    ###  calculate moving / rolling averages from specified window sizes

    for n in rolling_windows:
        if n < n_lags:
            for aggregate in rolling_aggregates:
                ts_data['TRS_' + str(n) + 'hr_lag_' + aggregate] = ts_data[["TRS_" + str(n) + "hr_lag_" + str(x) for x in range(1, n+1)]].agg([aggregate], axis=1)
                ts_data['weighted_SO2_' + str(n) + 'hr_lag_' + aggregate] = ts_data[["weighted_SO2_" + str(n) + "hr_lag_" + str(x) for x in range(1, n+1)]].agg([aggregate], axis=1)

    return ts_data



def decomposition_feature_engineering(ts_data, decompose_lags: int, deceompose_method: str, feature_name: str):

    assert deceompose_method in ["additive","multiplicative"], "error: method not unavailable"

    seasonality_dict = {}
    trend_dict = {}
    resid_dict = {}

    for n in ts_data['station'].drop_duplicates():
        decompose = seasonal_decompose(ts_data[ts_data['station'] == n].set_index('date_hour')[feature_name], 
                                       model = deceompose_method, 
                                       period = decompose_lags, 
                                       extrapolate_trend=True)
        seasonality_dict[n] = decompose.seasonal
        trend_dict[n] = decompose.trend
        resid_dict[n] = decompose.resid

    seasonality = pd.melt(pd.DataFrame(seasonality_dict).reset_index(), id_vars="date_hour")
    trend = pd.melt(pd.DataFrame(trend_dict).reset_index(), id_vars="date_hour")
    resid = pd.melt(pd.DataFrame(resid_dict).reset_index(), id_vars="date_hour")

    seasonality.columns = ['date_hour','station', deceompose_method + '_seasonality_' + feature_name]
    trend.columns = ['date_hour','station', deceompose_method + '_trend_' + feature_name]
    resid.columns = ['date_hour','station', deceompose_method + '_resid_' + feature_name]

    return [seasonality, trend, resid]






if __name__ == "__main__":

    ts_data, train_stations, all_stations = get_training_data()
    station_distances = get_station_distance_matrix(all_stations)
    ts_data = process_training_features(ts_data, start_time = "2021-12-31", end_time = "2023-12-31")
    ts_data = inverse_distance_weighted_so2(train_stations, station_distances, ts_data)
    ts_data = lag_feature_engineering(ts_data, n_lags=36)

    decompose_trs = decomposition_feature_engineering(ts_data, 
                                                      decompose_lags = 24, 
                                                      deceompose_method = "additive", 
                                                      feature_name = "TRS")

    decompose_weighted_so2 = decomposition_feature_engineering(ts_data, 
                                                               decompose_lags = 24, 
                                                               deceompose_method = "additive", 
                                                               feature_name = "weighted_SO2")

    ts_data = reduce(lambda df1, df2: pd.merge(df1, df2, on=['date_hour','station'], how="left"), 
                              [ts_data,
                               decompose_trs[0], decompose_trs[1], decompose_trs[2],
                               decompose_weighted_so2[0], decompose_weighted_so2[1], decompose_weighted_so2[2],
                               ])
    
    decompose_trs = decomposition_feature_engineering(ts_data, 
                                                      decompose_lags = 24, 
                                                      deceompose_method = "multiplicative", 
                                                      feature_name = "TRS")

    decompose_weighted_so2 = decomposition_feature_engineering(ts_data, 
                                                               decompose_lags = 24, 
                                                               deceompose_method = "multiplicative", 
                                                               feature_name = "weighted_SO2")

    ts_data = reduce(lambda df1, df2: pd.merge(df1, df2, on=['date_hour','station'], how="left"), 
                              [ts_data,
                               decompose_trs[0], decompose_trs[1], decompose_trs[2],
                               decompose_weighted_so2[0], decompose_weighted_so2[1], decompose_weighted_so2[2],
                               ])

    ts_data = ts_data.dropna()    
    ts_data['SO2_above_0'] = ts_data['SO2'].apply(lambda x: 0 if x == 0 else 1)
