import os

import pandas as pd

from src.helper_functions.estimation.estimate_mixed_logit import standardize


def load_comscore_categories():
    categories = pd.read_csv("data/comscore/input/comscore_categories.csv", dtype=str)[
        "Category_Code"
    ].tolist()
    print(f"Categories needed: {categories}")
    print(f"Number of categories needed: {len(categories)}")

    return categories


def load_data_long_comscore(input_path, save_long_data=False):
    # Paths to the relevant data
    choice_data_path = os.path.join(input_path, "long_format_data.csv")
    asin_list_path = os.path.join(input_path, "asin_to_item_crosswalk.csv")
    principal_components_path = os.path.join(input_path, "principal_components")

    # Step 1: Load asin list as strings
    asin_list = pd.read_csv(asin_list_path, dtype={"asin_code": str})
    product_id_to_asin = dict(zip(asin_list["product_id"], asin_list["asin_code"]))
    sorted_asins = sorted(asin_list["asin_code"])

    # Step 2: Load principal components into a dictionary
    principal_components = {}
    for file in os.listdir(principal_components_path):
        if file.endswith("_principal_components.csv"):
            model = file.split("_principal_components")[0]
            filepath = os.path.join(principal_components_path, file)

            # Load the CSV and assign the sorted asin as index
            pc_data = pd.read_csv(filepath)

            # Only keep the asins that are in the asin_list and use the sorted asin order
            pc_data = pc_data[pc_data["asin"].isin(sorted_asins)]
            pc_data.index = sorted_asins  # Use sorted asin order
            pc_data.drop(columns=["asin"], inplace=True)

            # Assert that pc_data has the same number of rows as asin_list and has no NaN values
            assert pc_data.shape[0] == len(sorted_asins)

            # Add each principal component column to the dictionary
            for i, col in enumerate(pc_data.columns, start=1):
                # Normalize each principal component
                pc_normalized = standardize(pc_data[col])
                principal_components[f"{model}_pc{i}"] = dict(
                    zip(pc_data.index, pc_normalized)
                )

    # Step 3: Load choice data
    choice_data = pd.read_csv(choice_data_path)

    # Rename columns to match the estimation script
    choice_data.rename(
        columns={"purchase": "choice", "order_id": "choice_id"}, inplace=True
    )

    # Sort the choice data by choice_id and then by product_id for consistent ordering
    choice_data.sort_values(by=["choice_id", "product_id"], inplace=True)

    # Group by choice_id and count the number of products for each choice_id
    product_counts = choice_data.groupby("choice_id").size()

    # Check if all choice_ids have the same number of products
    assert (
        product_counts.nunique() == 1
    ), "Not all choice_ids have the same number of products."

    # Assertion 2: Ensure that each choice_id has one and only one choice
    # Group by choice_id and sum the choice column (should be 1 for each group)
    choice_sums = choice_data.groupby("choice_id")["choice"].sum()
    assert all(
        choice_sums == 1
    ), "Each choice_id should have exactly one choice (choice == 1)."

    # Check that the rest of the choices are 0
    # This is implied by the previous assertion, but you can also explicitly check it
    choice_data_grouped = choice_data.groupby("choice_id")["choice"].apply(
        lambda x: list(x)
    )
    for choices in choice_data_grouped:
        assert (
            choices.count(1) == 1 and choices.count(0) == len(choices) - 1
        ), "Each choice_id should have exactly one choice == 1 and the rest choice == 0."

    # Build product fixed effects columns
    # first, create a list of product_ids (unique values in the product_id column), then for each, create a column 'product_id_{product_id}' with 1 if the product_id matches the current product_id and 0 otherwise
    product_ids = choice_data["product_id"].unique()
    product_fixed_effects = {
        f"product_id_{product_id}": (choice_data["product_id"] == product_id).astype(
            int
        )
        for product_id in product_ids
    }

    # Add the product fixed effects to choice_data
    for colname, coldata in product_fixed_effects.items():
        choice_data[colname] = coldata

    # Step 4: Merge principal components into choice_data

    # Map product_id to asin for choice_data
    choice_data["asin"] = choice_data["product_id"].map(
        lambda x: product_id_to_asin.get(x)
    )

    # Add columns for each principal component
    for key, pc_dict in principal_components.items():
        choice_data[key] = choice_data["asin"].map(pc_dict)

    # Assert that nothing is NaN
    assert (
        not choice_data.isnull().values.any()
    ), f"Columns with NaN: {choice_data.columns[choice_data.isnull().any()]}"

    # Save the long data to a CSV
    if save_long_data:
        output_path = os.path.join(input_path, "long_format_data_with_pc.csv")
        choice_data.to_csv(output_path, index=False)
        print(f"Long data saved to: {output_path}")

    return choice_data
