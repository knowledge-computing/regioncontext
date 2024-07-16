import logging
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pandas as pd
import geopandas as gpd
from shapely import Point
from shapely import wkt
import geohash
import numpy as np
import h3 

from polygeohasher import polygeohasher
import geopandas as gpd
from polygon_geohasher.polygon_geohasher import polygon_to_geohashes, geohashes_to_polygon


from regioncontext.model_trainer._utils.helpers import geohash_to_polygon, cell_to_shapely
from regioncontext.utils import const

import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
import geopandas as gpd

def read_shapefile_to_csv(input_shapefile_path, columns_to_keep, output_csv_path):
    try:
        # Check if the shapefile exists
        if not input_shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found at {input_shapefile_path}")
        
        # Read the shapefile using geopandas
        shapefile = gpd.read_file(input_shapefile_path)
        print(shapefile.head(2))
        # Convert the geometry to WGS84
        shapefile = shapefile.to_crs("EPSG:4326")

        #shapefile['geometry'] = shapefile['geometry'].apply(lambda geom: geom.wkt)
        # Loop through the columns in the shapefile
        for column in shapefile.columns:
            # Check if the column is in the dictionary
            if column not in columns_to_keep and column != 'geometry':
                # Delete the column from the shapefile
                shapefile.drop(column, axis=1, inplace=True)
        # Save the shapefile to CSV
        shapefile.to_csv(output_csv_path, index=False)
        return shapefile
    except FileNotFoundError as e:
        print(e)

def clean_csv(input_csv_path, src_column, output_csv_path):
    try:
        # Check if the shapefile exists
        if not input_csv_path.exists():
            raise FileNotFoundError(f"File not found at {input_csv_path}")
        # Read the CSV using pandas
        csv_data = pd.read_csv(input_csv_path)
        
        # Loop through each row in the CSV
        for index, row in csv_data.iterrows():
            # Check if the row[column] string starts with 'R1'
            if row[src_column].startswith('R') == True:
                csv_data.at[index, src_column] = 'Residential Districts'
            elif row[src_column].startswith('C')  == True:
                # Update the value in the src_column to a new value
                csv_data.at[index, src_column] = 'Commercial Districts'
            elif row[src_column].startswith('M1') and row[src_column].str.contains('R') == False:
                # Update the value in the src_column to a new value
                csv_data.at[index, src_column] = 'Manufacturing Districts'
            elif row[src_column].startswith('M1') and row[src_column].str.contains('R') == True:
                # Update the value in the src_column to a new value
                csv_data.at[index, src_column] = 'Mixed Manufacturing & Residential Districts'    
            elif row[src_column].startswith('BPC'):
                    # Update the value in the src_column to a new value
                    csv_data.at[index, src_column] = 'Battery Park City'   
            elif row[src_column].startswith('PARK'):
                    # Update the value in the src_column to a new value
                    csv_data.at[index, src_column] = 'PARK, BALL FIELD, PLAYGROUND and PUBLIC SPACE'   
        # Save the CSV to the output path
        csv_data.to_csv(output_csv_path, index=False)
       
    except FileNotFoundError as e:
        print(e)

def generate_aoi_region(in_csv_file_path, out_csv_file_path, region_type='geohash', region_level='10'):
        # Add your code here to generate the JSON
        try:
            df = pd.read_csv(in_csv_file_path)
            df[const.regioncontext_geometry_field_name] = df[const.regioncontext_geometry_field_name].apply(wkt.loads)
            gdf = gpd.GeoDataFrame(df, geometry=const.regioncontext_geometry_field_name, crs="EPSG:4326")
            initial_df = ''
            if region_type == 'geohash':
                pgh = polygeohasher.Polygeohasher(gdf)

                # create a dataframe with list of geohashes for each geometry
                initial_df = pgh.create_geohash_list(region_level,inner=False)
                # print(initial_df.columns)
                # initial_df = initial_df.rename(columns={'field_1': 'gid','field_2': 'osm_id','field_3': 'code','field_4': 'fclass','field_5': 'name'})

                # print(initial_df.columns)
                # make a row for each column in the geohash_list
                initial_df = initial_df.explode('geohash_list')
                initial_df['geohash_list'] = initial_df['geohash_list'].astype(str)
                initial_df['geometry'] = initial_df['geohash_list'].apply(lambda x: geohash.decode(str(x)))
                initial_df['geometry'] = initial_df['geometry'].apply(lambda x: ', '.join(str(x).split(', ')[::-1]))
                initial_df['geometry'] = 'Point (' + initial_df['geometry'].astype(str).str.replace('(', '').str.replace(')', '').str.replace(',', '')+')'
            else:
                 #todo: support for h3
                 pass
            initial_df.to_csv(out_csv_file_path, index=False)
            
        except Exception as e:
            logging.error(f"Error generating grid data: {e}")
        return None
        
def main():
    # Define the input shapefile path
    # input_shapefile_path = Path("/home/yaoyi/projects/RegionContext/data/nyc/sample/building.shp")
    
    # # Define the columns to keep
    # #columns_to_keep = ["fclass", "name"]
    # #columns_to_keep = ["ZoneDist1"]
    # #columns_to_keep = ['BoroName', 'NTAName']
    # columns_to_keep = ['type']
    # # Define the output CSV path
    # output_csv_path = Path("/home/yaoyi/projects/RegionContext/data/nyc/sample/building-sample.csv")
    
    # # Call the read_shapefile_to_csv function
    # read_shapefile_to_csv(input_shapefile_path, columns_to_keep, output_csv_path)
    # #clean_csv(input_shapefile_path, "ZoneDist1", output_csv_path)


    generate_aoi_region(in_csv_file_path='/home/yaoyi/projects/RegionContext/data/nyc/sample/landuse-sample.csv', 
                        out_csv_file_path='/home/yaoyi/projects/RegionContext/data/nyc/sample/aoi-landuse-sample.csv', region_type='geohash', region_level=8)


if __name__ == "__main__":
    main()