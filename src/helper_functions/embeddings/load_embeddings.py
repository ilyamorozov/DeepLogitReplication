import os

import numpy as np


def load_all_embeddings(image_embeddings_path, text_embeddings_path):
    """
    Loads all image and text embeddings from .npy files.

    Parameters
    ----------
    image_embeddings_path : str
        Path containing .npy files storing image embeddings.
    text_embeddings_path : str
        Path containing .npy files storing text embeddings.

    Returns
    -------
    dict
        Dictionary of (J, num_features) matrices where the key is the model.
    """
    embeddings = {}

    # Load image embeddings
    for file_name in os.listdir(image_embeddings_path):
        if file_name.endswith("_features.npy"):
            model_name = file_name.replace("_features.npy", "")
            file_path = os.path.join(image_embeddings_path, file_name)
            embeddings[f"{model_name}_image"] = np.load(file_path)
            if embeddings[f"{model_name}_image"].shape[1] == 0:
                raise ValueError(
                    f"Number of features for {model_name} image embeddings is 0. \n {image_embeddings_path}"
                )

    # Load text embeddings
    for file_name in os.listdir(text_embeddings_path):
        if file_name.endswith("_features.npy"):
            model_name = file_name.replace("_features.npy", "")
            file_path = os.path.join(text_embeddings_path, file_name)
            embeddings[f"{model_name}_text"] = np.load(file_path)
            if embeddings[f"{model_name}_text"].shape[1] == 0:
                raise ValueError(
                    f"Number of features for {model_name} text embeddings is 0. \n {text_embeddings_path}"
                )

    # Validate that the first dimension (number of rows) matches across all matrices
    num_samples = {key: matrix.shape[0] for key, matrix in embeddings.items()}
    if len(set(num_samples.values())) > 1:
        raise ValueError(
            f"Inconsistent number of samples across embeddings: {num_samples}"
        )

    return embeddings
