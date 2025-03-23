# python -m src.replicate_comscore.1_comscore_generate_embeddings

import os
from glob import glob

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

from src.helper_functions.embeddings.generate_embeddings import (
    generate_image_embeddings,
    generate_text_embeddings,
)


def load_product_images(dir_path, asins):
    """Loads product texts and images from the given file path.

    Parameters
    -------
    dir_path : str
        Directory path containing book texts and images.
    asins : list
        List of ASINs we expect to find data for

    Returns
    -------
    images : dict
        Dictionary with ASINs as keys and preprocessed image arrays as values.
    """

    # Load book images
    images = {}
    image_paths = glob(os.path.join(dir_path, "images", "*.png"))
    target_size = (224, 224)

    # use .jpg as a fallback
    if not image_paths:
        image_paths = glob(os.path.join(dir_path, "images", "*.jpg"))
        target_size = (224, 224)

    if not image_paths:
        raise ValueError(
            f"No images found in directory {dir_path} with extension *.png or *.jpg"
        )

    # print(f"Found {len(image_paths)} images in {dir_path}")

    # Check for missing and extra ASINs
    image_asins = {os.path.splitext(os.path.basename(path))[0] for path in image_paths}
    asins_set = set(asins)
    missing_asins = asins_set - image_asins
    extra_asins = image_asins - asins_set

    if missing_asins:
        print(f"Directory: {dir_path}")
        print(
            f"Field 'image' has missing values for {len(asins) - len(image_asins)} ASINs"
        )
        raise ValueError("Missing images")
    if extra_asins:
        print(f"Directory: {dir_path}")
        print(f"Number of extra ASINs: {len(extra_asins)}")  # ASINs: {extra_asins}")

    for asin in asins:
        if asin in image_asins:
            image_path = os.path.join(dir_path, "images", f"{asin}.png")
            try:
                img = image.load_img(image_path, target_size=target_size)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                images[asin] = img_array
            except Exception as e:
                image_path = os.path.join(dir_path, "images", f"{asin}.jpg")
                try:
                    img = image.load_img(image_path, target_size=target_size)
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    images[asin] = img_array
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
        else:
            # Create a blank white image as a placeholder
            blank_image = (
                np.ones((target_size[0], target_size[1], 3), dtype=np.uint8) * 255
            )
            img_array = np.expand_dims(blank_image, axis=0)
            images[asin] = img_array

    images = dict(sorted(images.items()))

    # Assert that we have loaded len(asins) images
    assert len(images) == len(asins)

    # Assert that all images are the same shape
    shapes = {img.shape for img in images.values()}
    if len(shapes) != 1:
        print(f"Error: Images have different shapes")
        print(f"Asins: {asins}")
        print(f"Directory: {dir_path}")
        print(f"Image shapes: {shapes}")
    assert len(shapes) == 1

    return images


def load_product_texts(dir_path, asins):
    """Loads product texts and images from the given file path.

    Parameters
    -------
    dir_path : str
        Directory path containing book texts and images.
    asins : list
        List of ASINs we expect to find data for

    Returns
    -------
    texts : dict
        Dictionary with ASINs as keys and text data as a dictionary as values.
    """
    number_of_reviews = 5

    def read_csv_with_fallback(file_path, encodings=None):
        """
        Reads a CSV file with a fallback mechanism for different encodings.
        """
        if encodings is None:
            encodings = ["utf-8", "latin1", "iso-8859-1", "windows-1252"]

        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                print(f"Encoding {encoding} failed.")
            except Exception as e:
                print(f"An error occurred with encoding {encoding}: {e}")

        raise ValueError("All encodings failed. Unable to read the file.")

    # Load product description CSV
    if "product_descriptions.csv" in os.listdir(dir_path):
        products = read_csv_with_fallback(
            os.path.join(dir_path, "product_descriptions.csv")
        )
    else:
        # try description.csv as a fallback
        products = read_csv_with_fallback(os.path.join(dir_path, "descriptions.csv"))

    # Combine all description columns into a single description column
    max_index = 20
    description_columns = ["product description"] + [
        str(i) for i in range(max_index + 1) if str(i) in products.columns
    ]
    if "description" in products.columns:
        description_columns.append("description")

    products["description_real"] = products[description_columns].apply(
        lambda x: " ".join(x.dropna()), axis=1
    )

    # Drop the combined columns from the dataframe
    products = products.drop(columns=description_columns)
    products = products.rename(columns={"description_real": "description"})

    # Load reviews CSV
    reviews = pd.read_csv(os.path.join(dir_path, "reviews.csv"))

    if "product_title" not in products.columns:
        products = products.merge(
            reviews[["asin", "product_title"]].drop_duplicates(), on="asin", how="left"
        )

    reviews = reviews.drop(columns=["product_title"])
    reviews["review_number"] = reviews.groupby("asin").cumcount() + 1
    reviews_filtered = reviews[reviews["review_number"] <= number_of_reviews]
    reviews_pivot = reviews_filtered.pivot(
        index=["asin"], columns="review_number", values="review_text"
    )
    reviews_pivot = reviews_pivot.rename(columns=lambda x: f"user_review_{x}")
    reviews_pivot = reviews_pivot.reset_index()

    # Merge reviews with products
    products = products.merge(reviews_pivot, on="asin", how="left")
    products = products.rename(columns={"product_title": "title"})
    products = products.fillna("")

    # Check for missing and extra ASINs
    product_asins = set(products["asin"])
    asins_set = set(asins)
    missing_asins = asins_set - product_asins
    extra_asins = product_asins - asins_set

    if missing_asins:
        print(f"Missing {len(asins)} - {len(product_asins)} ASINs: {missing_asins}")
        raise ValueError("Missing texts")
    if extra_asins:
        print(f"Directory: {dir_path}")
        print(f"Number of extra ASINs: {len(extra_asins)}")  # ASINs: {extra_asins}")

    # Load product texts dictionary
    texts = {}

    for asin in asins:
        if asin in product_asins:
            i = products["asin"].tolist().index(asin)
            texts[asin] = {
                "title": products["title"][i],
                "description": products["description"][i],
            }
            for j in range(number_of_reviews):
                texts[asin][f"user_review_{j+1}"] = products[f"user_review_{j+1}"][i]
        else:
            texts[asin] = {
                "title": "",
                "description": "",
            }
            for j in range(number_of_reviews):
                texts[asin][f"user_review_{j+1}"] = ""

    # Check for any missing ("") values in texts
    missing_fields = {"title": [], "description": []}
    for j in range(number_of_reviews):
        missing_fields[f"user_review_{j+1}"] = []

    for asin, fields in texts.items():
        for field, value in fields.items():
            if value == "":
                missing_fields[field].append(asin)

    for field, asins_with_missing in missing_fields.items():
        if asins_with_missing:
            print(
                f"Field '{field}' has missing values for {len(asins_with_missing)} ASINs"
            )

    texts = dict(sorted(texts.items()))

    assert len(texts) == len(asins)

    return texts


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
                # read asin as string
                asin_df = pd.read_csv(asin_list_path, dtype=str)
                asin_list = sorted(asin_df["asin"].tolist())
                category_to_asins[category] = asin_list
            else:
                raise ValueError(f"File 'asin_list.csv' not found in {category_path}")
    return category_to_asins


def main():
    """Generates image and text embeddings for each model and saves them to disk."""

    category_to_asins = load_category_to_asins("data/comscore/input/selected_products/")
    categories_needed = pd.read_csv(
        "data/comscore/input/comscore_categories.csv", dtype=str
    )["Category_Code"].tolist()

    for category in categories_needed:
        asins = category_to_asins[category]
        # Set file paths
        input_dir_path = f"data/comscore/input/text_and_images/{category}/"
        intermediate_dir_path = f"data/comscore/intermediate/{category}/embeddings/"
        image_dir_path = os.path.join(intermediate_dir_path, "images/")
        text_dir_path = os.path.join(intermediate_dir_path, "texts/")

        # Load book text and images
        images = load_product_images(input_dir_path, asins)
        texts = load_product_texts(input_dir_path, asins)
        assert list(images.keys()) == list(texts.keys())

        # Generate embeddings for each model and save
        generate_image_embeddings(images, image_dir_path)
        generate_text_embeddings(texts, text_dir_path)


if __name__ == "__main__":
    main()
