import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rioxr
import xarray as xr
import exactextract as eet
import shapely

from functools import reduce



## locate the main folder path
main_folder = f""


# target grids

with open(f'/src/reference_grids.json', 'r') as f:
    coords = json.load(f)

lon_centroids = coords[0]
lat_centroids = coords[1]

grid = gpd.read_file(f'/src/reference_grids.shp')
grid = grid.set_crs(epsg=4326)



##  recall spatial time series tables
#       2021 only covering december records for generating lagged features for the beginning days in 2022 Jan
#       2022 and 2023 are full-year records

def read_gee_exported_csv(file_list, 
                          year_range = [2021, 2022, 2023]):

    output = list ()

    for n in year_range:

        fusion_year = list()

        for n in range(len([x for x in file_list if n in x])):

            tem = pd.read_csv(main_folder + [x for x in file_list if n in x][n])
            tem = tem.iloc[1:,:]
            tem['time'] = tem["id"].apply(lambda x: str(x)[0:4] + "-" + str(x)[4:6] + "-" + str(x)[6:])
            tem['longitude'] = tem['longitude'].apply(lambda x: float(x))
            tem['latitude'] = tem['latitude'].apply(lambda x: float(x))
            tem = tem.drop(["id"], axis=1)
            
            fusion_year.append(tem)

        fusion = reduce(lambda x, y: pd.merge(x, y, 
                                                on = ['longitude', 'latitude', 'time'], 
                                                how='outer'), 
                                                fusion_year)
        
        output.append(fusion)

    output = pd.concat(output).reset_index(drop=True)
    output_xr = xr.Dataset.from_dataframe(output.set_index(['longitude','latitude','time']))

    return output_xr


##  interpolation to project the feature arrays to the specified target grid resolution

def reproj_sts_array_to_target_grids(xr_array):

    xr_correct = xr_array.rename({"longitude": "lon", "latitude": "lat"}).astype(float)
    xr_correct['lat'] = xr_correct['lat'].astype(float)
    xr_correct['lon'] = xr_correct['lon'].astype(float)
    xr_correct['time'] = pd.DatetimeIndex(xr_correct['time'])

    #  interpolation with linear spline
    xr_interp = xr_correct.interp(lat = lat_centroids, 
                                  lon = lon_centroids, 
                                  method='slinear')

    return xr_interp


def zonal_stats_raster_to_target_grids(tiff_source: str, 
                                       extract_statistic: list):

    feature = rioxr.open_rasterio(tiff_source)

    out_dict = {}

    # mean & std for continuous raster, e.g. DEM, building height, population density

    if "mean" in extract_statistic:
        feature_mean = eet.exact_extract(feature, grid, 'mean')
        extract_mean_list = [feature_mean[n]['properties']['mean'] for n in range(grid.shape[0])]
        out_dict["mean"] = pd.Series(extract_mean_list)

    if "stdev" in extract_statistic:    
        feature_std = eet.exact_extract(feature, grid, 'stdev')
        extract_std_list = [feature_std[n]['properties']['stdev'] for n in range(grid.shape[0])]
        out_dict["stdev"] = pd.Series(extract_std_list)

    # fractions per class for categorical raster, e.g. Land Use / Land Cover

    if "frac" in extract_statistic:    
        feature_frac = eet.exact_extract(feature, grid, 'frac')
        out_dict["frac"] = feature_frac

    if "unique" in extract_statistic:    
        feature_unique_values = eet.exact_extract(feature, grid, 'unique')
        out_dict["unique"] = feature_unique_values

    if "frac" in extract_statistic and "unique" in extract_statistic:
        out_list = []

        for n in range(grid.shape[0]):
            for m in range(len(feature_frac[n]['properties']['frac'])):
                grid_id = grid['id'].iloc[n]
                class_val = int(feature_unique_values[n]['properties']['unique'][m])
                class_frac = feature_frac[n]['properties']['frac'][m]
                out_ = pd.DataFrame({'id': grid_id, 'class': class_val, 'frac': class_frac}, index=[0])
                out_list.append(out_)

        out_df = pd.concat(out_list).reset_index(drop=True)
        out_df = pd.pivot_table(out_df, index=['id'], columns=['class'], values=['frac']).reset_index()
        out_df = out_df.fillna(0)

        out_dict["class_frac"] = out_df

    return out_dict



def zonal_stats_vector_to_target_grids(shapefile_source: str, 
                                       feature_type: str, 
                                       id_column: str,
                                       summarize_column: str):

    assert feature_type in["point", "line"], NotImplementedError

    feature = gpd.read_file(shapefile_source)
    feature = feature.to_crs(epsg=4326)

    if feature_type == "point":

        feature_agg = gpd.sjoin(feature, 
                                grid, 
                                how="inner", 
                                predicate='within') \
                        .groupby([id_column])    \
                        .agg({summarize_column: "count"}) \
                        .reset_index()
        
    elif feature_type == "line":

        feature_agg = gpd.overlay(grid, 
                                  feature, 
                                  how="intersection", 
                                  keep_geom_type=False)
        
        feature_agg["length"] = feature_agg.geometry.length

        feature_agg = feature_agg.groupby([id_column]) \
                                .agg({summarize_column: "count",
                                      "length": "sum"}) \
                                .reset_index()

    return feature_agg

