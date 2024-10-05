from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar

from pysal.explore import esda
from pysal.lib import weights
from shapely import Polygon
from shapely import Point

import contextily
import geopandas as gpd
import numpy as np



def spatial_weights(plot_data, spatial_weight_method: str, knn_neighbours=None):

    assert spatial_weight_method in ["Queen","Rook","KNN"], NotImplementedError

    if knn_neighbours:
        assert isinstance(knn_neighbours, int), NotImplementedError

    if spatial_weight_method == "Queen":
        plot_weights = weights.Rook.from_dataframe(plot_data)

    return plot_weights



def getis_ord_gi_star(spatial_weight, plot_data, plot_attribute):

    go_hotspot = esda.getisord.G_Local(plot_data[plot_attribute], 
                                       spatial_weight, 
                                       star=True, 
                                       transform='R', 
                                       permutations=1000, 
                                       seed=123)
    
    return go_hotspot



def hotspot_plot(plot_data, getis_ord_gi_star_hotspot, geo_boundary, point_data):

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    plt.rcParams['hatch.linewidth'] = 3

    ec = "0.5"

    db = gpd.GeoDataFrame(plot_data, crs="epsg:4326")

    ##  1)  Getis-Ord Gi* hotspot
    # Break observations into significant or not
    sig = getis_ord_gi_star_hotspot.p_sim < 0.05
    # Plot HH clusters
    hh1 = db.loc[(getis_ord_gi_star_hotspot.Zs > 0) & (sig == True), "geometry"]
    hh1.plot(ax=ax, color="coral", edgecolor=ec, linewidth=0.1, alpha=0.4, facecolor="#000000", hatch="//", label = "Getis-Ord Gi*", aspect=None)

    ##  2)  90-th percentile hotspot
    # Break observations into significant or not
    top_quintile = np.percentile(db["mean_so2"], 90)
    low_quintile = np.percentile(db["mean_so2"], 10)
    sig = (db["mean_so2"] > top_quintile) | (db["mean_so2"] < low_quintile)
    # Plot clusters
    hh2 = db.loc[(db["mean_so2"] > top_quintile) & (sig == True), "geometry"]
    hh2.plot(ax=ax, color="slateblue", edgecolor=ec, linewidth=0.1, alpha=0.4, facecolor="#000000", hatch="o", label = "90-th Percentile", aspect=None)

    ##  3) add Saint John, or any boundary
    gpd.GeoDataFrame(geo_boundary).plot(ax=ax, facecolor="none", edgecolor='black', linewidth=0.3, aspect=None)

    #   4) add station locations
    point_data.plot(ax=ax, color="deepskyblue", markersize=100, edgecolor='black', linewidth=1, aspect=None)
    
    ##  5) add basemap
    contextily.add_basemap(
        ax=ax,
        crs="epsg:4326",
        source=contextily.providers.CartoDB.Positron,
    )

    return fig, ax



def create_scale_bar(ax, geo_boundary):

    ##  adjust boundary
    bounds = geo_boundary.geometry.bounds
    x0, x1, y0, y1 = bounds.minx, bounds.maxx, bounds.miny, bounds.maxy

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ##  specify 1 degree of distance for creating a spatial scale bar
    points = gpd.GeoSeries(
        [Point(x0, y0), Point(x1, y0)], crs=4326
    )
    points = points.to_crs(3347)
    scale_meters = points[0].distance(points[1])
    ax.add_artist(ScaleBar(scale_meters, font_properties={"size": 18}))



def create_legends(ax, ec, mode):
    
    assert mode in ["polygon","line","point"], NotImplementedError

    if mode == "polygon":

        ## add legend
        lines = [
            Patch(facecolor=t.get_facecolor(), hatch=t.get_hatch(), edgecolor=ec)
            for t in ax.collections[0:-1]
        ]
        labels = [t.get_label() for t in ax.collections[0:-1]]
        ax.legend(lines, labels, loc="lower right", prop={'size': 18})

    else:

        ## add legend
        lines = [
            Line2D([0], [0], linestyle="none", marker="s", markersize=14, markerfacecolor=t.get_facecolor())
            for t in ax.collections[0:-1]
        ]
        labels = [t.get_label() for t in ax.collections[0:-3]]
        legend = ax.legend(lines, labels, bbox_to_anchor=(0.6,0), loc="lower right", prop={'size': 18})

        ax.add_artist(legend)
