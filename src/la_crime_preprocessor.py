"""
la crime preproceesor joins the crimes from 2 different sources into a single file. 
It combines data from  Los Angeles police department and  Los Angeles sheriff department
"""
# import modules
import os
import pandas as pd
import config


# lapd = Los Angeles police department
# lasd = Los Angeles sheriff department

# lapd only the City of Los Angeles (LAPD),
# while the lasd covers unincorporated areas and contract cities within the County of Los Angeles (LASD).

lapd_path = config.PATHS["lapd"]
lasd_path = config.PATHS["lasd"]

# read crime from lapd file and select only required columns
lapd_df = pd.read_csv(lapd_path)
lapd_df = lapd_df[["DR_NO", "LAT", "LON"]]

# read crime from lasd file and select only required columns
lasd_df = pd.read_csv(lasd_path)
lasd_df = lasd_df[["LURN_SAK", "LATITUDE", "LONGITUDE"]]


# rename columns to combine 2 sources of data
lapd_df = lapd_df.rename(columns = {
    "DR_NO" : "id",     
})

lasd_df = lasd_df.rename(columns = {
    "LURN_SAK" : "id",  
    "LATITUDE" : "LAT", 
    "LONGITUDE" :  "LON"
})

# write to a file
final_df = pd.concat([lapd_df, lasd_df])
final_df.to_csv(os.path.join(config.PROJECT_ROOT, "data/raw/crime/LA_crime.csv"), index=False)