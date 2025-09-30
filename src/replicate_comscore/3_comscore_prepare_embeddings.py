# python -m src.replicate_comscore.2_comscore_prepare_embeddings

import os

import pandas as pd

from src.helper_functions.embeddings.compute_principal_components import (
    compute_principal_components,
)
from src.helper_functions.embeddings.load_embeddings import load_all_embeddings
from src.helper_functions.file_structure.get_file_path_from_config import get_file_path_from_config


def load_category_to_asins(dir_path):
    """Loads category to ASINs mapping from the given file path.

    Parameters
    -------
    dir_path : str
        Directory path containing category to ASINs mapping.

    Returns
    -------
    category_to_asins : dict
        Dictionary with category names as keys and list of ASINs as values.
    """
    category_to_asins = {}
    for category in os.listdir(dir_path):
        category_path = os.path.join(dir_path, category)
        if os.path.isdir(category_path):
            asin_list_path = os.path.join(category_path, "asin_list.csv")
            if os.path.isfile(asin_list_path):
                asin_list = sorted(pd.read_csv(asin_list_path)["asin"].tolist())
                category_to_asins[category] = asin_list
            else:
                print(f"ASIN list not found for category: {category}")
    return category_to_asins


def main():
    input_path = get_file_path_from_config(path_type="COMSCORE_3", path="INPUT_PATH")
    intermediate_path = get_file_path_from_config(path_type="COMSCORE_3", path="INTERMEDIATE_PATH")
    category_to_asins = load_category_to_asins(os.path.join(input_path, "selected_products"))
    print(f"Categories: {category_to_asins.keys()}")
    print(f"Number of categories: {len(category_to_asins)}")
    categories_needed = pd.read_csv(
        os.path.join(input_path, "comscore_categories.csv"), dtype=str
    )["Category_Code"].tolist()
    print(f"Categories needed: {categories_needed}")
    print(f"Number of categories needed: {len(categories_needed)}")

    for category in categories_needed:
        asins = category_to_asins[category]
        # Set directory paths
        category_path = os.path.join(intermediate_path, category)
        embeddings_path = os.path.join(category_path, "embeddings/")

        # Input directories
        image_embeddings_path = os.path.join(embeddings_path, "images/")
        text_embeddings_path = os.path.join(embeddings_path, "texts/")

        # Output directories
        principal_components_path = os.path.join(
            category_path, "principal_components/"
        )

        # Load image and text embeddings and compute principal components
        embeddings = load_all_embeddings(image_embeddings_path, text_embeddings_path)
        compute_principal_components(embeddings, principal_components_path, asins)


if __name__ == "__main__":
    main()
