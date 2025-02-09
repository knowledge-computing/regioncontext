import logging
import os, sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import Point, Polygon
from shapely import wkt, wkb

import geohash
from polygeohasher import polygeohasher
from polygon_geohasher.polygon_geohasher import polygon_to_geohashes, geohashes_to_polygon

import h3.api.basic_str as h3
import h3pandas

from model_trainer._utils.helpers import geohash_to_polygon, cell_to_shapely
from utils import const


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



def generate_aoi_region(in_csv_file_path, out_csv_file_path, context_field_names, region_type, region_level):
        try:
            df = pd.read_csv(in_csv_file_path)
            df[const.regioncontext_geometry_field_name] = df[const.regioncontext_geometry_field_name].apply(wkt.loads)
            
            df[const.aggregated_field_name]=''
            for field_name in context_field_names:
                df[field_name] = df[field_name].astype(str).apply(lambda x: x.lower().encode('ascii', 'ignore').strip().decode('ascii') if x != 'nan' else np.nan)
                df[const.aggregated_field_name] = df[const.aggregated_field_name] + df[field_name] + ':'
            
            df = df[df[const.aggregated_field_name].notnull()]
            df[const.aggregated_field_name] = df[const.aggregated_field_name].apply(lambda x: x[:-1])
            
            df = df.reset_index(drop=True)

            gdf = gpd.GeoDataFrame(df, geometry=const.regioncontext_geometry_field_name, crs="EPSG:4326")

            initial_df = pd.DataFrame()
            
            if region_type == 'geohash':
                pgh = polygeohasher.Polygeohasher(gdf)
                initial_df = pgh.create_geohash_list(region_level, inner=False)
                initial_df = initial_df.explode('geohash_list')

                initial_df[const.regioncontext_geometry_field_name] = initial_df['geohash_list'].apply(lambda x: geohashes_to_polygon(x))
                
                def geohash_to_point(gh):
                    lat, lon = geohash.decode(gh)
                    return Point(lon, lat)  # (lon, lat) order
                
                initial_df[const.regioncontext_geometry_field_name] = initial_df['geohash_list'].astype(str).apply(geohash_to_point)

            elif region_type == 'h3':
                gdf = gdf.explode(index_parts=False)
                gdf = gdf.reset_index(drop=True)
                initial_df = gdf.h3.polyfill(region_level, explode=True)
                initial_df = initial_df.rename(columns={'h3_polyfill': 'h3_list'})
                initial_df = initial_df.dropna(subset=['h3_list'], ignore_index=True)
                initial_df[const.regioncontext_geometry_field_name] = initial_df['h3_list'].apply(lambda x: Point(*h3.h3_to_geo(x)[::-1]))

            initial_df[const.poi_aoi_field_name] = const.aoi_field_value
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_path', type=str, required=True)
    parser.add_argument('--output_csv_path', type=str, required=True)
    
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    generate_aoi_region(in_csv_file_path=args.input_csv_path, out_csv_file_path=args.output_csv_path, context_field_names = ['subtype','class'], region_type=const.regioncontext_region_type, region_level=const.regioncontext_region_level)


if __name__ == "__main__":
    main()