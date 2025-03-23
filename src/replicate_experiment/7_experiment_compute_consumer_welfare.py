# python -m src.replicate_experiment.7_experiment_compute_consumer_welfare

import itertools
import os
import time

import numpy as np
import pandas as pd

from src.helper_functions.estimation.estimate_mixed_logit import load_data_long
from src.helper_functions.estimation.load_model_results import (
    generate_util_matrix,
    load_coeff_dicts,
)


def main():
    """
    1) Load each of the three models from the Excel file.
       - We look up the row in the relevant sheet where 'Specification' matches.
       - We parse 'Coefficient Names' and 'Estimated Coefficients' into a dict.

    2) Using each model's coeff_dict, compute an N×J utility matrix for all consumers/products.

    3) Enumerate 2^J possible choice sets and compute consumer welfare as E[max(u_j)] across
       the included products for each consumer. Average across consumers.

    4) Store results in a DataFrame with columns for each model's welfare and 10 indicator columns.
    """
    # ------------------------------
    # 0. Load Data
    # ------------------------------
    input_path = "data/experiment/input"
    first_choice_data, _, empirical_diversion_matrix, book_titles = load_data_long(
        input_path
    )

    date = "2025-03-21"
    estimation_results_path = (
        f"data/experiment/output/estimation_results/mixed_logit_results_{date}.xlsx"
    )

    # "observables-plain logit" -> sheet "observables", specification "plain logit"
    # "observables-pages and year" -> sheet "observables", specification "pages and year"
    # "user_reviews_USE_text-PC1 and PC2" -> sheet "user_reviews_USE_text", specification "PC1 and PC2"
    models_to_use = [
        "observables-plain logit",
        "observables-pages and year",
        "user_reviews_USE_text-PC1 and PC2",
    ]

    # Map each model to a column name in final CSV
    model_column_map = {
        "observables-plain logit": "CS_Logit",
        "observables-pages and year": "CS_X",
        "user_reviews_USE_text-PC1 and PC2": "CS_PCA",
    }

    # ------------------------------
    # 1. Load Excel sheets & parse model coefficients
    # ------------------------------
    all_sheets = pd.read_excel(estimation_results_path, sheet_name=None)

    # For storing {model_str: coeff_dict}
    model_coeff_dicts = {}

    for model_str in models_to_use:
        model_coeff_dicts[model_str] = load_coeff_dicts(all_sheets, model_str)

    # ------------------------------
    # 2. Compute utility (N×J) for each model
    # ------------------------------
    choice_ids = first_choice_data["choice_id"].unique()
    unique_products = first_choice_data["product_id"].unique()
    J = len(unique_products)

    # We'll store each model's N×J matrix in a dict
    model_utilities = {}

    # Some fixed offset for random seeds
    seed_offset = 123

    for model_str, coeff_dict in model_coeff_dicts.items():
        util_matrix, rmse = generate_util_matrix(
            first_choice_data,
            coeff_dict,
            seed_offset=seed_offset,
            empirical_diversion_matrix=empirical_diversion_matrix,
        )

        print(f"RMSE for {model_str}: {rmse:.4f}")
        model_utilities[model_str] = util_matrix

    # ------------------------------
    # 3. Enumerate all 2^J subsets and compute ex ante consumer welfare
    # ------------------------------
    subsets = list(itertools.product([0, 1], repeat=J))
    print(f"Number of subsets: {len(subsets)}")

    # Build DataFrame with 10 columns for inclusion indicators
    subset_df = pd.DataFrame(subsets, columns=book_titles)

    # Add columns for each model's welfare
    for model_str in models_to_use:
        col = model_column_map[model_str]
        subset_df[col] = 0.0

    # For each subset, compute E[max_{included}(utility)] across consumers
    start_t = time.time()
    for idx, subset_tuple in enumerate(subsets):
        included_mask = np.array(subset_tuple, dtype=bool)

        # Skip if no products are included
        if not included_mask.any():
            continue

        for model_str in models_to_use:
            col = model_column_map[model_str]
            util_mat = model_utilities[model_str]  # shape (N, J)
            subset_util = util_mat[:, included_mask]  # shape (N, #included)
            # max across included products
            max_util = np.max(subset_util, axis=1)
            # average across consumers
            welfare = np.mean(max_util)
            subset_df.at[idx, col] = welfare

    elapsed = time.time() - start_t
    print(f"Computed welfare for all subsets/models in {elapsed:.2f}s")

    # ------------------------------
    # 4. Save to CSV
    # ------------------------------
    output_dir = "data/experiment/output/consumer_welfare"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"consumer_welfare_{date}.csv")
    subset_df.to_csv(output_path, index=False)

    print(f"Consumer welfare results saved to: {output_path}")


if __name__ == "__main__":
    main()
