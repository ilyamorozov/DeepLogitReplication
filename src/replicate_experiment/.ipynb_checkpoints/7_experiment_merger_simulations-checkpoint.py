import numpy as np
import pandas as pd

from src.helper_functions.estimation.estimate_mixed_logit import load_data_long
from src.helper_functions.estimation.load_model_results import (
    generate_util_matrix,
    load_coeff_dicts,
)


def market_shares(prices, positions, model, first_choice_data, book_titles):
    """
    Compute market shares for all books given prices, positions, and a model.

    Args:
        prices (list): List of prices for each book (length 10).
        positions (list): List of positions (ranks) for each book (length 10).
        model (dict): Coefficient dictionary for the model.
        first_choice_data (pd.DataFrame): DataFrame containing consumer choices.
        book_titles (list): List of book titles.

    Returns:
        list: List of tuples containing book index, book title, and market share.
    """
    choice_ids = first_choice_data["choice_id"].unique()
    unique_products = first_choice_data["product_id"].unique()
    N = len(choice_ids)
    J = len(unique_products)
    assert len(prices) == len(positions) == J
    assert len(first_choice_data) // J == N

    # Update the price and position columns in first_choice_data
    first_choice_data["price"] = np.tile(prices, len(first_choice_data) // len(prices))
    first_choice_data["position"] = np.tile(
        positions, len(first_choice_data) // len(positions)
    )

    # Compute utilities
    util_matrix = generate_util_matrix(first_choice_data, model)

    # Convert utilities to probabilities
    exp_util = np.exp(util_matrix)
    sum_exp_util = exp_util.sum(axis=1, keepdims=True)
    probabilities = exp_util / sum_exp_util

    # Assert that probabilities sum to 1 for each consumer
    assert np.allclose(probabilities.sum(axis=1), 1)

    # Calculate market shares by averaging probabilities across consumers
    market_shares = probabilities.mean(axis=0)

    # Pair market shares with book indices and titles
    result = list(zip(range(1, J + 1), book_titles, market_shares))

    return result


def main():

    # ------------------------------
    # Load data and model coefficients
    # ------------------------------

    input_path = "data/experiment/input"
    first_choice_data, _, _, book_titles = load_data_long(input_path)

    date = "2025-01-28"
    estimation_results_path = (
        f"data/experiment/output/estimation_results/mixed_logit_results_{date}.xlsx"
    )
    all_sheets = pd.read_excel(estimation_results_path, sheet_name=None)

    # Define models to use
    models_to_use = [
        "observables-plain logit",
        "observables-pages and year",
        "user_reviews_USE_text-PC1 and PC2",
    ]

    # Load coefficient dictionaries for each model
    model_coeff_dicts = {}
    for model_str in models_to_use:
        model_coeff_dicts[model_str] = load_coeff_dicts(all_sheets, model_str)

    # ------------------------------
    # Market shares example
    # ------------------------------

    # Example prices and positions
    prices = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Compute market shares for each model
    for model_str, coeff_dict in model_coeff_dicts.items():
        print(f"Computing market shares for model: {model_str}")

        # Example usage of market_shares function
        shares = market_shares(
            prices, positions, coeff_dict, first_choice_data, book_titles
        )

        for book_index, title, share in shares:
            print(f"Book {book_index} ({title}): {share:.4f}")
        print("\n")


if __name__ == "__main__":
    main()
