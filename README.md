# RegionContext

RegionContext is a Python-based project designed to analyze and cluster Points of Interest (POIs) based on spatial and contextual similarities. Utilizing advanced machine learning techniques, including SpaBERT and KMeans clustering, the project aims to provide insights into regional patterns and relationships among POIs.

## Features

- **SpaBERT Prediction**: Leverages the SpaBERT model to predict contextual embeddings for POIs based on their descriptions and categories.
- **Autoencoder Dimensionality Reduction**: Applies an autoencoder to reduce the dimensionality of the contextual embeddings, facilitating more efficient clustering.
- **KMeans Clustering**: Implements KMeans clustering to group POIs into clusters based on their reduced-dimensional embeddings, allowing for the analysis of regional similarities and differences.
- **Flexible Data Handling**: Supports input and output in various formats, including CSV and JSON, for easy integration with other tools and workflows.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python packages: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `keras`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RegionContext.git