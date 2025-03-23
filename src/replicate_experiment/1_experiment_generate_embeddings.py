# python -m src.replicate_experiment.1_experiment_generate_embeddings

import os
from glob import glob

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

from src.helper_functions.embeddings.generate_embeddings import (
    generate_image_embeddings,
    generate_text_embeddings,
)


def load_book_texts_and_images(dir_path):
    """Loads book texts and images from the given file path.

    Returns
    -------
    images : dict
        Dictionary with ASINs as keys and preprocessed image arrays as values.
    texts : dict
        Dictionary with ASINs as keys and text data as a dictionary as values.
    """

    # Load book images
    images = {}
    image_paths = glob(os.path.join(dir_path, "book_covers", "*.jpg"))
    target_size = (224, 224)

    if not image_paths:
        raise ValueError(
            f"No images found in directory {dir_path} with extension *.png"
        )

    print(f"Found {len(image_paths)} images.")

    for path in image_paths:
        path = path.replace("\\", "/")
        asin = os.path.splitext(os.path.basename(path))[0]
        try:
            img = image.load_img(path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            images[asin] = img_array
        except Exception as e:
            print(f"Error loading image {path}: {e}")

    images = dict(sorted(images.items()))

    # Load book texts
    texts = {}

    books = pd.read_csv(os.path.join(dir_path, "books.csv"))

    required_columns = [
        "asin",
        "title",
        "description",
        "user_review_1",
        "user_review_2",
        "user_review_3",
        "user_review_4",
        "user_review_5",
    ]

    if not all(col in books.columns for col in required_columns):
        raise ValueError(f"books.csv must contain columns: {required_columns}")
    if books["asin"].duplicated().any():
        raise ValueError("Duplicate ASINs found in books.csv")

    for i in range(len(books)):
        asin = books["asin"][i]
        texts[asin] = {
            "title": books["title"][i],
            "description": books["description"][i],
            "user_review_1": books["user_review_1"][i],
            "user_review_2": books["user_review_2"][i],
            "user_review_3": books["user_review_3"][i],
            "user_review_4": books["user_review_4"][i],
            "user_review_5": books["user_review_5"][i],
        }

    # TODO: validate text data, check for missing values

    texts = dict(sorted(texts.items()))

    # Check that images and texts have the same ASINs
    assert list(images.keys()) == list(texts.keys())

    return images, texts


def main():
    """Generates image and text embeddings for each model and saves them to disk."""

    # Set file paths
    input_dir_path = "data/experiment/input/books/"
    intermediate_dir_path = "data/experiment/intermediate/embeddings/"
    image_dir_path = os.path.join(intermediate_dir_path, "images/")
    text_dir_path = os.path.join(intermediate_dir_path, "texts/")

    # Load book text and images
    images, texts = load_book_texts_and_images(input_dir_path)

    # Generate embeddings for each model and save
    generate_image_embeddings(images, image_dir_path)
    generate_text_embeddings(texts, text_dir_path)


if __name__ == "__main__":
    main()
