import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

### retrieve the census subdivision area of Saint John

csd_shp = gpd.read_file(f"./src/lcsd000b21a_e.shp")
csd_sj = csd_shp[csd_shp["CSDNAME"] == "Saint John"]
csd_sj_geom_gjs = gpd.GeoDataFrame(csd_sj[["geometry"]])
csd_sj_geom_gjs = csd_sj_geom_gjs.set_crs(epsg=3347)
csd_sj_geom_gjs = csd_sj_geom_gjs.to_crs(epsg=4326)


def generate_output_grid(degree_resolution: float|int,
                         output_type: str):

    ## calculate image dimensions
    width = np.array((csd_sj_geom_gjs.bounds.maxx - csd_sj_geom_gjs.bounds.minx) / degree_resolution)[0]
    height = np.array((csd_sj_geom_gjs.bounds.maxy - csd_sj_geom_gjs.bounds.miny) / degree_resolution)[0]

    ##  consider there are two grid-line edges per each dimension
    dim = (int(height) + 2, int(width) + 2)

    ##  starting from northend and westend
    top_left_corner_h = np.array([csd_sj_geom_gjs.bounds.minx + degree_resolution * z for z in range(dim[0])]).reshape(-1).tolist()
    top_right_corner_h = top_left_corner_h[1:] + [top_left_corner_h[-1] + degree_resolution]
    top_left_corner_v = np.array([csd_sj_geom_gjs.bounds.maxy - degree_resolution * z for z in range(dim[1])]).reshape(-1).tolist()
    bottom_left_corner_v = top_left_corner_v[1:] + [top_left_corner_v[-1] - degree_resolution]

    centroids_h = top_left_corner_h + [top_right_corner_h[-1]]
    centroids_v = top_left_corner_v + [bottom_left_corner_v[-1]]
    centroids_h = [0.5 * (centroids_h[x] + centroids_h[x+1]) for x in range(len(centroids_h) - 1)]
    centroids_v = [0.5 * (centroids_v[x] + centroids_v[x+1]) for x in range(len(centroids_v) - 1)]

    coords = [
        centroids_h,
        centroids_v
    ]

    polys_coords = list()
    polys_ids = list()

    for i in range(dim[0]):
        for j in range(dim[1]):
            polys_coords.append(
                shapely.Polygon([
                    [top_left_corner_h[i], top_left_corner_v[j]],
                    [top_right_corner_h[i], top_left_corner_v[j]],
                    [top_right_corner_h[i], bottom_left_corner_v[j]],
                    [top_left_corner_h[i], bottom_left_corner_v[j]],
                    [top_left_corner_h[i], top_left_corner_v[j]],
                ])
            )
            polys_ids.append(str(i) + "_" + str(j))

    grids_coords = gpd.GeoDataFrame({
        "id": polys_ids,
        "geometry": polys_coords
    })

    if output_type == "json":
        return coords
    elif output_type == "geodataframe":
        return grids_coords
    else:
        return NotImplementedError
    

