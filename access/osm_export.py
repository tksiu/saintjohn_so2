import osmnx

##  OpenStreetMap features
PLACE_NAME = "Saint John, Canada"

##  buildings
buildings = osmnx.geometries_from_place(
    PLACE_NAME,
    {"building": True},
)

## highways
highways = osmnx.geometries_from_place(
    PLACE_NAME,
    {"highway": True},
)
highways = highways[highways["highway"].isin(["motorway","trunk","primary","secondary","tertiary"])]


