import pandas as pd
import argparse
from pathlib import Path
import pandas as pd
import json
from shapely import wkt
import scipy.spatial as scp

from utils import const 

class GenerateSpaBERTJSON:
    def __init__(
            self, 
            verbose: bool = False,):
        self.verbose = verbose
        self.__df = None

    def fit_transform(self, csv_file_path, context_field_names, geometry_field_name,
            num_neighbors, 
            search_radius_meters, pseudo_sentence_json_file_path):
        self.generate_json(csv_file_path, context_field_names, geometry_field_name,
            num_neighbors, 
            search_radius_meters)
        self.save_json(pseudo_sentence_json_file_path)

    def generate_json(self, csv_file_path, context_field_names, geometry_field_name,
            num_neighbors=100, 
            search_radius_meters=100 ) -> pd.DataFrame:
        self.csv_file_path = csv_file_path
        self.context_field_names = context_field_names
        self.geometry_field_name = geometry_field_name
        self.num_neighbors = num_neighbors
        self.search_radius_meters = search_radius_meters
        df = pd.read_csv(self.csv_file_path)
        df[const.aggregated_field_name]=''
        for field_name in self.context_field_names:
            df[field_name] = df[field_name].apply(lambda x: str(x).lower().encode('ascii', 'ignore').strip().decode('ascii'))
            df[const.aggregated_field_name] = df[const.aggregated_field_name]+df[field_name] + ':'
        df[const.aggregated_field_name] = df[const.aggregated_field_name].apply(lambda x: x[:-1])

        df = df[df[const.aggregated_field_name].notnull()]
        df = df[df[self.geometry_field_name].notnull()]
        df = df.reset_index(drop=True)

        df[self.geometry_field_name] = df[self.geometry_field_name].apply(wkt.loads)
        df[const.x_y_field_name] = df[self.geometry_field_name].apply(lambda x: [x.y, x.x])
        df[const.psuedo_sentence_field_name] =''
        ordered_neighbor_coordinate_list = scp.KDTree(df[const.x_y_field_name].values.tolist())

        for index, row in df.iterrows():
            if const.poi_aoi_field_name in row:
                if row[const.poi_aoi_field_name] == const.aoi_field_value:
                    continue
            nearest_dist, nearest_neighbors_idx = ordered_neighbor_coordinate_list.query([row[const.x_y_field_name]], k=self.num_neighbors+1)
            # Loop through nearest_neighbors_idx and remove elements based on some condition
            filtered_neighbors_idx = []
            filtered_nearest_dist = []
            idx = 0
            for i in nearest_dist[0]:
                if float(i) < float(self.search_radius_meters/const.wgs842meters):   #100 meters
                    filtered_neighbors_idx.append(
                        nearest_neighbors_idx[0][idx])
                    filtered_nearest_dist.append(i)
                idx += 1

            nearest_neighbors_context = []
            nearest_neighbors_coords = []
            nearest_neighbors_dist = []
            idx = 0

            for i in filtered_neighbors_idx:
                neighbor_context = df[const.aggregated_field_name][i]
                neighbor_coords = df[const.x_y_field_name][i]
                nearest_neighbors_context.append(neighbor_context)
                nearest_neighbors_coords.append(
                    {const.neighbor_coordinates: neighbor_coords})
                nearest_neighbors_dist.append(
                    {const.neighbor_distance: float(filtered_nearest_dist[idx])*const.wgs842meters})
                idx += 1
            if self.verbose:
                print(f"Processing row {index+1}/{len(df)}")
            neighbor_info = {const.neighbor_context_list: nearest_neighbors_context,
                            const.neighbor_geometry_list: nearest_neighbors_coords, const.neighbor_distance: nearest_neighbors_dist}
            ps = {const.psuedo_sentence_info: {const.pivot_id: index+1, const.pivot_context: row[const.aggregated_field_name], const.pivot_geometry: {
                    const.pivot_coordinates: row[const.x_y_field_name]}}, const.neighbor_info: neighbor_info}
            df.at[index, const.psuedo_sentence_field_name] = ps
        if self.verbose:
            print(df.head(2))
        self.__df = df[df[const.psuedo_sentence_field_name]!=''][[const.psuedo_sentence_field_name]]
        return self.__df
    
    def save_json(self, pseudo_sentence_json_file_path):
        with open(pseudo_sentence_json_file_path, 'w') as out_f:
            for index, row in self.__df.iterrows():
                if const.poi_aoi_field_name in row:
                    if row[const.poi_aoi_field_name] == const.aoi_field_value:
                        continue
                out_f.write(json.dumps(row[const.psuedo_sentence_field_name]))
                out_f.write('\n')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_path', type=str)
    parser.add_argument('--num_neighbors', type=int)
    parser.add_argument('--search_radius_meters', type=int)

    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    current_file_path = Path(__file__).resolve()
    current_dir_path = current_file_path.parent
    city = 'irbid'
    csv_file_name = 'novateur-poi-sample.csv'
    csv_file_path = current_dir_path.parent / 'data' / city / csv_file_name

    generator = GenerateSpaBERTJSON(csv_file_name=csv_file_name, data_dir_path=Path(current_dir_path.parent / 'data'))
    generator.generate_json()
    
    generator.save_json()

if __name__ == '__main__':
    main()
