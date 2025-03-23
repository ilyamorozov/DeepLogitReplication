import os

import numpy as np
from scipy.spatial.distance import cdist


def compute_pairwise_distances(embeddings, distances_path, metric="euclidean"):
    """
    Compute pairwise distances for each embedding and save as .npy files.

    Parameters
    ----------
    embeddings : dict
        Dictionary of (J, num_features) matrices where the key is the model.
    distances_path : str
        Directory where the distance matrices will be saved.
    metric : str
        Distance metric to use, default is "euclidean". Can be any metric
        supported by `scipy.spatial.distance.cdist`.
    """
    # Ensure the output directory exists
    os.makedirs(distances_path, exist_ok=True)

    for model, matrix in embeddings.items():

        # Compute the (J, J) distance matrix
        distances = cdist(matrix, matrix, metric=metric)  # type: ignore

        # Save the distances to a file
        file_name = f"{model}_distances.npy"
        file_path = os.path.join(distances_path, file_name)
        np.save(file_path, distances)

        print(
            f"Saved pairwise distances for {model} to {file_path}, shape = {distances.shape}"
        )
