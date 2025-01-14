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

    go_gi_stats = esda.getisord.G_Local(plot_data[plot_attribute], 
                                       spatial_weight, 
                                       star=True, 
                                       transform='R', 
                                       permutations=1000, 
                                       seed=123)

    sig = go_gi_stats.p_sim < 0.05
    highs = go_gi_stats > 0

    hotspot = plot_data.loc[(sig & highs), "geometry"]
    
    return hotspot



def top_decile(plot_data, plot_attribute):
    
    top_decile = np.percentile(plot_data[plot_attribute], 90)
    hotspot = plot_data.loc[plot_data[plot_attribute] > top_decile, "geometry"]

    return hotspot



def create_basemap(ax, basemap=contextily.providers.CartoDB.PositronNoLabels, zoom_level=14):
    
    contextily.add_basemap(
        ax=ax,
        crs="epsg:4326",
        source=basemap,
        zoom=zoom_level
    )



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
    ax.add_artist(ScaleBar(scale_meters, font_properties={"size": 18}, box_alpha=0.5))



def create_legends(ax, ec, mode, elem_index_start, elem_index_end, pos="best", pos_adjust_x=1, pos_adjust_y=0):
    
    assert mode in ["filled polygon","line","point"], NotImplementedError

    if mode == "polygon":

        ## add legend
        lines = [
            Patch(facecolor=t.get_facecolor(), hatch=t.get_hatch(), edgecolor=ec)
            for t in ax.collections[elem_index_start:elem_index_end]
        ]
        labels = [t.get_label() for t in ax.collections[elem_index_start:elem_index_end]]
        ax.legend(lines, labels, loc=pos, prop={'size': 18})

    else:

        ## add legend
        lines = [
            Line2D([0], [0], linestyle="none", marker="s", markersize=14, markerfacecolor=t.get_facecolor())
            for t in ax.collections[elem_index_start:elem_index_end]
        ]
        labels = [t.get_label() for t in ax.collections[elem_index_start:elem_index_end]]
        legend = ax.legend(lines, labels, bbox_to_anchor=(pos_adjust_x, pos_adjust_y), loc=pos, prop={'size': 18})

        ax.add_artist(legend)
