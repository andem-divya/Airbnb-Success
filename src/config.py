
"""
config file
"""
import os

CRS = "EPSG:4326"

# List of cities
CITIES = ["OAK", "SF", "LA", "SD"]

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to project

# Paths for data
PATHS = {
    "crime": {city: os.path.join(PROJECT_ROOT, f"data/raw/crime/{city}_crime.csv") for city in CITIES},
    "airbnb": {city: os.path.join(PROJECT_ROOT,f"data/raw/airbnb/listings_{city}.csv") for city in CITIES},
    "census": os.path.join(PROJECT_ROOT, "data/raw/census_tract/tl_2025_06_tract/tl_2025_06_tract.shp"),
    "lapd": os.path.join(PROJECT_ROOT, "data/raw/crime/LAPD_crime.csv"),
    "lasd": os.path.join(PROJECT_ROOT, "data/raw/crime/LASD_crime.csv"),
    "population": os.path.join(PROJECT_ROOT, "data/raw/population/ACSDT5Y2023.B01003-Data.csv"),
    "walkability": os.path.join(PROJECT_ROOT, "data/raw/walkability/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv")
}

# Column name mapping for each city
COL_MAP = {
    "crime": {
        "OAK": {"lat": None, "long": None, "wkt": "Location"},
        "SF": {"lat": "Latitude", "long": "Longitude", "wkt": None},
        "LA": {"lat": "LAT", "long": "LON", "wkt": None},
        "SD": {"lat": "latitude", "long": "longitude", "wkt": None},
        "LAPD": {"lat": "LAT", "long": "LON", "wkt": None},
        "LASD": {"lat": "LATITUDE", "long": "LONGITUDE", "wkt": None}
    }
}
