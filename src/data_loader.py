# import modules

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
from src import config
import numpy as np
from sklearn.neighbors import NearestNeighbors

def airbnb_data_load(path, lat_col, long_col):
    """
    loads airbnb listings data

    Args:
        path (str): path of the data file to read.
        lat_col (str): name of the latitude column.
        long_col (str): name of the longitude column.

    Returns:
        gdf: geometric dataframe
    """
    df = pd.read_csv(path)
    geometry = gpd.points_from_xy(df[long_col], df[lat_col])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=config.CRS)
    return gdf

def population_data_load(path, columns_to_select=["GEOID", "B01003_001E"]):
    """
    loads population data

    Args:
        path (str): path of the data file to read.
        columns_to_select : list of columns to select in the df

    Returns:
        df: pandas dataframe
    """
    # skip one line after header
    df = pd.read_csv(path, header=0, skiprows=[1])

    # Strip the prefix to match TIGER shapefile format (06001400100)
    df['GEOID'] = df['GEO_ID'].str.replace('1400000US', '')

    # select only required columns
    if columns_to_select:
     df = df[columns_to_select]

    # rename column
    df = df.rename(columns={"B01003_001E":"population"})

    return df

def walkability_data_load(path, columns_to_select=["GEOID", "NatWalkInd"]):
    """
    loads walkability scores

    Args:
        path (str): path of the data file to read.
        columns_to_select : list of columns to select in the df

    Returns:
        df: pandas dataframe
    """
    # read_csv
    df = pd.read_csv(path)

    # select state, county, tract 
    state = df['STATEFP'].astype(str).str.zfill(2)
    county = df['COUNTYFP'].astype(str).str.zfill(3)
    tract = df['TRACTCE'].astype(str).str.zfill(6)

    # create GEOID from abouve values
    geoid_series = state + county + tract

    # add new geioid series to our original df as a column
    df = df.copy()
    df["GEOID"] = geoid_series

    if columns_to_select:
        return df[columns_to_select]

    return df

def crime_data_load(path, lat_col, long_col, wkt_col):
    """
    loads crime data

    Args:
        path (str): path of the data file to read.
        lat_col (str): name of the latitude column.
        long_col (str): name of the longitude column.
        wkt_col(str): name of the geometry shape column

    Returns:
        gdf: geometric dataframe
    """
    df = pd.read_csv(path)

    def parse_point(val):
        if pd.isna(val):
            return None
        return wkt.loads(str(val))
    
    if wkt_col:
        geometry = df[wkt_col].apply(parse_point)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=geometry,
            crs=config.CRS)

    elif lat_col and long_col:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[long_col], df[lat_col]),
            crs=config.CRS
        )

    else:
        raise ValueError(
            "Provide either wkt_col or both lat_col and lon_col"
        )

    return gdf

def assign_census_tract_id(points_gdf, census_gdf, tract_id_col="GEOID", geometry_col="geometry", area_col="ALAND"):
    """
    Assigns census tract IDs to a GeoDataFrame of points using spatial join.

    Args:
        points_gdf (GeoDataFrame): GeoDataFrame with point geometries.
        census_gdf (GeoDataFrame): GeoDataFrame of census tracts with polygons.
        tract_id_col (str): Column name in census_gdf that contains the tract ID.
        area_col (str): Column name in census_gdf that contains the Area in square meters.


    Returns:
        GeoDataFrame: points_gdf with an added column GEOID.
    """
    # Defensive CRS check
    if points_gdf.crs != census_gdf.crs:
        census_gdf = census_gdf.to_crs(points_gdf.crs)

    # Spatial join
    gdf_with_tracts = gpd.sjoin(points_gdf, census_gdf[[tract_id_col, geometry_col, area_col]], how="left", predicate="within")
    
    # Convert area of the land (ALAND) to sq miles
    gdf_with_tracts['area_sqmiles'] = gdf_with_tracts['ALAND'] * 0.0000003861

    return gdf_with_tracts


def agg_by_census_tract(gdf, tract_id_col='GEOID'):
    """
    Aggregates data by census_tract

    Args:
        gdf (GeoDataFrame): GeoDataFrame.
        tract_id_col (str): Column name in gdf that contains the census tract ID.

    Returns:
        df: df with tract_id_col and count
    """

    df = (
        gdf
        .groupby(tract_id_col)
        .size()
        .reset_index(name="crime_count")
    )
    return df



# Downtown and airport coordinates
anchors = {
    "SF": {
        "downtown": (37.788056, -122.407500),   # Union Square
        "airport":  (37.61889,  -122.37500)     # SFO
    },
    "OAK": {
        "downtown": (37.80436, -122.27114),     # Jack London Square
        "airport":  (37.72139, -122.22083)      # OAK
    },
    "LA": {
        "downtown": (34.05000, -118.25000),     # Downtown LA
        "airport":  (33.94250, -118.40806)      # LAX
    },
    "SD": {
        "downtown": (32.72056, -117.15444),     # Downtown
        "airport":  (32.73361, -117.18972)      # SAN
    }
}

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance in kilometers.
    """
    R = 6371.0088  # Earth radius in km
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def add_location_distance_features(
    df,
    city_col="city",
    lat_col="latitude",
    lon_col="longitude"
):
    df = df.copy()
    
    # Initialize columns
    df["dist_km_downtown"] = np.nan
    df["dist_km_airport"] = np.nan
    
    for city, pts in anchors.items():
        mask = df[city_col] == city
        
        # Downtown distance
        df.loc[mask, "dist_km_downtown"] = haversine_km(
            df.loc[mask, lat_col].astype(float),
            df.loc[mask, lon_col].astype(float),
            pts["downtown"][0],
            pts["downtown"][1]
        )
        
        # Airport distance
        df.loc[mask, "dist_km_airport"] = haversine_km(
            df.loc[mask, lat_col].astype(float),
            df.loc[mask, lon_col].astype(float),
            pts["airport"][0],
            pts["airport"][1]
        )
    
    # Log versions (important for linear models)
    df["log_dist_km_downtown"] = np.log1p(df["dist_km_downtown"])
    df["log_dist_km_airport"] = np.log1p(df["dist_km_airport"])
    
    return df


# Distance-based imputation for cols_to_distance_impute
def impute_by_nearest_neighbor_efficient(df_in, target_col, lat_col='latitude', lon_col='longitude'):
    df_temp = df_in.copy()

    # Identify rows with and without missing values in the target column
    df_missing = df_temp[df_temp[target_col].isna()]
    df_present = df_temp[df_temp[target_col].notna()]
    
    if df_missing.empty or df_present.empty:
        return df_temp # No imputation needed or possible
    
    # Convert latitude and longitude to radians for haversine distance metric
    coords_present_rad = np.deg2rad(df_present[[lat_col, lon_col]].values)
    coords_missing_rad = np.deg2rad(df_missing[[lat_col, lon_col]].values)

    # Build NearestNeighbors model on points where target_col is present
    # Using 'ball_tree' for efficiency with haversine metric
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='haversine')
    nn.fit(coords_present_rad)

    # Find the nearest neighbor for each missing point
    # k-neighbors returns distances and indices
    distances, indices = nn.kneighbors(coords_missing_rad)

    # Impute the missing values
    # indices is 2D, e.g., [[0], [1], [2]] so we flatten it
    closest_neighbors_indices_in_present_df = indices.flatten()
    
    # Get the actual index from the original df_present to align values
    original_indices_of_closest_neighbors = df_present.iloc[closest_neighbors_indices_in_present_df].index

    # Map the imputed values back to the original DataFrame
    # Use .loc for safe assignment by index
    df_temp.loc[df_missing.index, target_col] = df_present.loc[original_indices_of_closest_neighbors, target_col].values
            
    return df_temp
