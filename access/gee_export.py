import ee
import numpy as np
import pandas as pd
import geopandas as gpd


##  initialize a GEE access token, put your Google Cloud project credential into the string
ee.Authenticate()
ee.Initialize(project='')


### retrieve the census subdivision area of Saint John

csd_shp = gpd.read_file(f"./src/lcsd000b21a_e.shp")
csd_sj = csd_shp[csd_shp["CSDNAME"] == "Saint John"]
csd_sj_geom_gjs = gpd.GeoDataFrame(csd_sj[["geometry"]])
csd_sj_geom_gjs = csd_sj_geom_gjs.set_crs(epsg=3347)
csd_sj_geom_gjs = csd_sj_geom_gjs.to_crs(epsg=4326)

### define a bounding box for accessing GEE data

region = ee.Geometry.Rectangle([np.min(csd_sj_geom_gjs.bounds.minx),
                                np.min(csd_sj_geom_gjs.bounds.miny),
                                np.max(csd_sj_geom_gjs.bounds.maxx),
                                np.max(csd_sj_geom_gjs.bounds.maxy)])


## export mapping function

def create_export(values, feature_names):
    return ee.Feature(None, ee.Dictionary.fromLists(feature_names, ee.List(values)))


## define a new function for calculating the percentage of non-null pixels

def image_perc_mask(image, scale, aoi):
    totPixels = ee.Number(image.unmask(1).reduceRegion(reducer = ee.Reducer.count(),
                                                       scale = scale,
                                                       geometry = aoi,
                                                       ).values().get(0))
    actPixels = ee.Number(image.reduceRegion(reducer = ee.Reducer.count(),
                                             scale = scale,
                                             geometry = aoi,
                                             ).values().get(0))
    percCover = actPixels.divide(totPixels).multiply(100).round()
    return image.set('percCover', percCover)


### export function

def gee_export_collection(
        feature_layers: list,
        collection: str,
        begin_date: str, 
        end_date: str,
        resolution: float|int,
        file_name_prefix: str,
        folder_name: str,
        daily_reduce=False
    ):
    
    if daily_reduce:
        date_range = pd.date_range(begin_date, end_date)

        day_avg = []

        for d in range(len(date_range) - 1):

            startdate = str(date_range[d]).split(" ")[0]
            enddate = str(date_range[d+1]).split(" ")[0]

            daily_extract = ee.ImageCollection(collection) \
                    .filterDate(startdate, enddate) \
                    .select(feature_layers) \
                    .filterBounds(region) \
                    .reduce(ee.Reducer.mean()) \
                    .set({'Start_Date': startdate,
                            'End_Date': enddate})

            day_avg.append(daily_extract)

        daily_collection = ee.ImageCollection.fromImages(day_avg)

    else:
        daily_collection = ee.ImageCollection(collection) \
                    .filterDate(startdate, enddate) \
                    .select(feature_layers) \
                    .filterBounds(region) \
                    .reduce(ee.Reducer.mean()) \
                    .set({'Start_Date': startdate,
                            'End_Date': enddate})

    for f in feature_layers:

        ## get data for each grid point
        tem_collection = daily_collection.filterBounds(region).select(f)
        ##  clipping collection to the CMA boundary
        tem_collection_non_null = tem_collection.map(
            lambda image: image_perc_mask(image, resolution, region)
            ).filter(ee.Filter.gt('percCover', 0)
        )

        ## processing
        tem_collection_array = tem_collection_non_null.getRegion(geometry = region, scale = resolution)
        feature_names = tem_collection_array.get(0)
        get_export = ee.FeatureCollection(
            tem_collection_array.map(
                lambda image: create_export(image, feature_names)
                )
            )

        ## exporting
        tasks = ee.batch.Export.table.toDrive(**{
                'collection': get_export,
                'description': file_name_prefix + f,
                'folder': folder_name,
                'fileFormat':'CSV',
                'selectors': ['id', 'longitude', 'latitude', 'time', 'Start_Date', 'End_Date', f]
        })
        tasks.start()

    
def gee_export_single_image(
        feature_layers: list,
        collection: str,
        begin_date: str, 
        end_date: str,
        resolution: float|int,
        file_name: str,
        folder_name: str,
        out_crs: str
    ):

    image_collection = ee.ImageCollection(collection) \
                        .filterDate(begin_date, end_date) \
                        .map(lambda x: x.clip(region)) \
                        .select(feature_layers) \
                        .reduce(ee.Reducer.mean())

    ## processing
    image = image_collection.toList(image_collection.size())
    image = ee.Image(image.get(0))

    tasks = ee.batch.Export.image.toDrive(**{
            'image': image,
            'description': file_name,
            'folder': folder_name,
            'region': region,
            'scale': resolution,
            'fileFormat':'GeoTIFF',
            'crs': out_crs
    })
    tasks.start()

