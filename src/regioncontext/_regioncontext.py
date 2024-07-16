
from data_processor.generate_spabert_json import GenerateSpaBERTJSON
from data_processor.generate_region_emb_csv import GenerateRegionEmbCSV
from model_trainer.spabert_trainer import SpaBERTTrainer
from dimension_reducer.autoencoder import AutoencoderReducer
from clustering.kmeans_clustering import KMeansClustering
import argparse
import utils as ut
import os
import sys
from pathlib import Path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


class RegionContext:
    def __init__(self):
        self.generate_spabert_json = GenerateSpaBERTJSON()
        pass

    def fit_transform(self, csv_file_path, context_field_names, geometry_field_name, num_neighbors, search_radius_meters):
        self.verbose = True
        parent_dir = Path(csv_file_path).resolve().parent
        file_name = Path(csv_file_path).resolve().stem
        data_dir = Path(parent_dir) / file_name
        if not data_dir.exists():
            data_dir.mkdir(parents=True)

        pseudo_sentence_json_file_path = data_dir / \
            f"{file_name}_pseudo_sentence_v2_nn{num_neighbors}_sdm{search_radius_meters}.json"
        model_weights_dir = data_dir / "model_weights"

        self.region = True
        self.region_type = 'h3'
        self.region_level = 10

        poi_embedding_csv_file_path = data_dir / \
            f"{Path(csv_file_path).resolve().stem}_emb_.csv"

        region_enc_embedding_csv_file_path = data_dir / \
            f"{Path(csv_file_path).resolve().stem}_emb_enc_{self.region_type}.{self.region_level}.csv"

        self.min_component = 3  # 10 - 50
        self.max_component = 4
        cluster_enc_embedding_csv_file_path = ''
        if self.min_component != self.max_component:
            cluster_enc_embedding_csv_file_path = data_dir / \
                f"{Path(csv_file_path).resolve().stem}_emb_enc_{self.region_type}.{self.region_level}_opt.csv"
        else:
            cluster_enc_embedding_csv_file_path = data_dir / \
                f"{Path(csv_file_path).resolve().stem}_emb_enc_{self.region_type}.{self.region_level}_{self.max_component}.csv"

        # # Generate the JSON file
        self.generate_spabert_json.fit_transform(csv_file_path=csv_file_path,
                                                 context_field_names=context_field_names,
                                                 geometry_field_name=geometry_field_name,
                                                 num_neighbors=num_neighbors,
                                                 search_radius_meters=search_radius_meters,
                                                 pseudo_sentence_json_file_path=pseudo_sentence_json_file_path)
        # Train the SpaBERT model
        spabert_trainer = SpaBERTTrainer()
        spabert_trainer.train_model(json_file_path=pseudo_sentence_json_file_path,
                                    model_save_dir=model_weights_dir,
                                    epochs=1, verbose=self.verbose)
        # # Predict the embeddings

        spabert_trainer.predict(json_file_path=pseudo_sentence_json_file_path,
                                model_save_dir=model_weights_dir,
                                csv_file_path=poi_embedding_csv_file_path, verbose=True)
        # # Generate the region embeddings
        generate_region_emb_csv = GenerateRegionEmbCSV()
        generate_region_emb_csv.fir_tranform(in_csv_file_path=poi_embedding_csv_file_path,
                                             out_csv_file_path=region_enc_embedding_csv_file_path,
                                             region_type=self.region_type, region_level=self.region_level)
        # Dimension reduction
        autoencoder_reducer = AutoencoderReducer()
        autoencoder_reducer.fit_transform(csv_file_path=poi_embedding_csv_file_path,
                                          enc_csv_file_path=region_enc_embedding_csv_file_path,
                                          epoch=300)
        # Clustering

        kmeans_clustering = KMeansClustering()
        kmeans_clustering.fit_predict(csv_file_path=region_enc_embedding_csv_file_path,
                                      output_csv_file_path=cluster_enc_embedding_csv_file_path,
                                      min_component=self.min_component, max_component=self.max_component, clustering_by_group=False, min_required_data_points=10, verbose=True)
        pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_path', type=str)
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    # Create an instance of RegionContext
    region_context = RegionContext()
    # args.input_csv_path
    csv_file_path = '~/projects/RegionContext/data/irbid/novateur-poi-sample.csv'

    # Call the fit_transform method
    region_context.fit_transform(
        csv_file_path=csv_file_path,
        context_field_names=['ps__category_level1',
                             'ps__category_level2', 'ta1_usage_descriptors'],
        # context_field_names=['poi_category'],
        geometry_field_name='WKT',
        num_neighbors=100, search_radius_meters=100)


if __name__ == "__main__":
    main()
