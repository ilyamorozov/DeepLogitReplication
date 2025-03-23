import json
import os

import numpy as np
import pandas as pd


def combine_embeddings(embeddings, combined_embeddings_path, asins):
    """
    Combine all feature matrices into a single matrix, save as .npy and .csv, and save column mapping as JSON.

    Parameters
    ----------
    embeddings : dict
        Dictionary of (J, num_features) matrices where the key is the model.
    combined_embeddings_path : str
        Path where the combined embeddings matrix and related files will be saved.
    asins : list
        List of ASINs corresponding to the rows in the embeddings.

    Returns
    -------
    combined_matrix : np.ndarray
        The combined feature matrix of shape (J, total_num_features).
    """
    # Ensure the output directory exists
    os.makedirs(combined_embeddings_path, exist_ok=True)

    # Verify that all embeddings have the same number of rows and match 'asins' length
    num_products = len(asins)
    for model, matrix in embeddings.items():
        if matrix.shape[0] != num_products:
            raise ValueError(
                f"Inconsistent number of products (rows) in embeddings for model '{model}'. "
                f"Expected {num_products} rows, got {matrix.shape[0]} rows."
            )

    # Sort models for consistent column mapping
    sorted_models = sorted(embeddings.keys())

    # Extract feature matrices in sorted order
    feature_matrices = [embeddings[model] for model in sorted_models]

    # Concatenate all feature matrices along the second axis (columns)
    combined_matrix = np.hstack(feature_matrices)

    # Save the combined matrix as all_features.npy
    npy_save_path = os.path.join(combined_embeddings_path, "all_features.npy")
    np.save(npy_save_path, combined_matrix)
    print(
        f"Combined embeddings saved to: {npy_save_path}, shape = {combined_matrix.shape}"
    )

    # Create a DataFrame for CSV, including 'asin'
    combined_df = pd.DataFrame(combined_matrix)
    combined_df.insert(0, "asin", asins)

    # Save as CSV
    csv_save_path = os.path.join(combined_embeddings_path, "all_features.csv")
    combined_df.to_csv(csv_save_path, index=False, float_format="%.18e")
    print(f"Combined embeddings saved to: {csv_save_path}")

    # Create column mapping
    column_mapping = {}
    current_col = 1  # Start after 'asin' which is at column 0
    for model in sorted_models:
        num_features = embeddings[model].shape[1]
        start_col = current_col
        end_col = current_col + num_features - 1
        column_mapping[model] = {"start_col": start_col, "end_col": end_col}
        current_col += num_features

    # Save column mapping as JSON
    json_save_path = os.path.join(combined_embeddings_path, "column_mapping.json")
    with open(json_save_path, "w") as f:
        json.dump(column_mapping, f, indent=4)
    print(f"Column mapping saved to: {json_save_path}")

    return combined_matrix
