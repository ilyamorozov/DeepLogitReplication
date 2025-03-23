# python -m src.replicate_experiment.2_experiment_prepare_embeddings

import os

import pandas as pd

from src.helper_functions.embeddings.compute_principal_components import (
    compute_principal_components,
)
from src.helper_functions.embeddings.load_embeddings import load_all_embeddings


def load_book_data(book_data_path):
    """Load observable book attributes"""
    return pd.read_csv(book_data_path)


def main():

    # Set directory paths
    intermediate_path = "data/experiment/intermediate/"
    embeddings_path = os.path.join(intermediate_path, "embeddings/")

    # Input directories
    image_embeddings_path = os.path.join(embeddings_path, "images/")
    text_embeddings_path = os.path.join(embeddings_path, "texts/")
    book_data_path = "data/experiment/input/books/books.csv"

    # Output directories
    principal_components_path = os.path.join(intermediate_path, "principal_components/")

    # Load image and text embeddings and book data
    embeddings = load_all_embeddings(image_embeddings_path, text_embeddings_path)
    book_data = load_book_data(book_data_path)
    asins = sorted(book_data["asin"].values.tolist())

    # Compute principal components for mixed logit
    compute_principal_components(embeddings, principal_components_path, asins)


if __name__ == "__main__":
    main()
