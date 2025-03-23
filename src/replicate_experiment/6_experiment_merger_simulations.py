import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
    util_matrix = generate_util_matrix(first_choice_data, model, include_gumbels=False)

    # Convert utilities to probabilities
    exp_util = np.exp(util_matrix)
    sum_exp_util = exp_util.sum(axis=1, keepdims=True)
    probabilities = exp_util / sum_exp_util

    # Assert that probabilities sum to 1 for each consumer
    assert np.allclose(probabilities.sum(axis=1), 1)

    # Calculate market shares by averaging probabilities across consumers
    # print(probabilities)
    market_shares = probabilities.mean(axis=0)

    # Pair market shares with book indices and titles
    result = list(zip(range(0, J), book_titles, market_shares))

    return result


def objective_function(
    prices,
    i,
    j,
    positions,
    coeff_dict,
    first_choice_data,
    book_titles,
    marginal_cost,
    price_other_books,
):
    """
    Computes the negative total profit given a pair of prices for books i and j.
    """
    prices_candidate = [price_other_books] * 10
    prices_candidate[i], prices_candidate[j] = prices[0], prices[1]

    shares = market_shares(
        prices_candidate, positions, coeff_dict, first_choice_data, book_titles
    )
    share_i = next(share for book_idx, _, share in shares if book_idx == i)
    share_j = next(share for book_idx, _, share in shares if book_idx == j)

    profit_i = (prices[0] - marginal_cost) * share_i
    profit_j = (prices[1] - marginal_cost) * share_j
    return -(profit_i + profit_j)  # Negative for minimization


def find_merged_optimum(
    price_grid,
    i,
    j,
    positions,
    coeff_dict,
    first_choice_data,
    book_titles,
    marginal_cost,
    price_other_books,
):
    """
    Finds the optimal prices for books i and j that maximize total profit.
    """
    initial_guess = [np.mean(price_grid), np.mean(price_grid)]
    bounds = [(min(price_grid), max(price_grid)), (min(price_grid), max(price_grid))]

    result = minimize(
        objective_function,
        initial_guess,
        args=(
            i,
            j,
            positions,
            coeff_dict,
            first_choice_data,
            book_titles,
            marginal_cost,
            price_other_books,
        ),
        bounds=bounds,
        method="L-BFGS-B",
    )

    best_price_i, best_price_j = result.x

    # Recalculate market shares at the optimal prices
    prices_optimal = [price_other_books] * 10
    prices_optimal[i] = best_price_i
    prices_optimal[j] = best_price_j

    shares = market_shares(
        prices_optimal, positions, coeff_dict, first_choice_data, book_titles
    )
    share_i = next(share for book_idx, _, share in shares if book_idx == i)
    share_j = next(share for book_idx, _, share in shares if book_idx == j)

    # Compute actual profits at the optimal prices
    profit_i = (best_price_i - marginal_cost) * share_i
    profit_j = (best_price_j - marginal_cost) * share_j
    total_profit = profit_i + profit_j

    return best_price_i, best_price_j, profit_i, profit_j, total_profit


# def find_merged_optimum(price_grid, i, j, positions, coeff_dict, first_choice_data, book_titles, marginal_cost, price_other_books):
#     """
#     Finds the optimal price combination that maximizes total profit when books i and j are merged.
#     """
#     results = []

#     for price_i in price_grid:
#         for price_j in price_grid:
#             prices_candidate = [price_other_books] * 10
#             prices_candidate[i] = price_i
#             prices_candidate[j] = price_j

#             shares = market_shares(prices_candidate, positions, coeff_dict, first_choice_data, book_titles)
#             share_i = next(share for book_idx, _, share in shares if book_idx == i)
#             share_j = next(share for book_idx, _, share in shares if book_idx == j)

#             profit_i = (price_i - marginal_cost) * share_i
#             profit_j = (price_j - marginal_cost) * share_j
#             total_profit = profit_i + profit_j

#             results.append((price_i, price_j, profit_i, profit_j, total_profit))

#     results_df = pd.DataFrame(results, columns=["Price_i", "Price_j", "Profit_i", "Profit_j", "Total_Profit"])
#     best_row = results_df.loc[results_df["Total_Profit"].idxmax()]
#     return best_row["Price_i"], best_row["Price_j"], best_row["Profit_i"], best_row["Profit_j"], best_row["Total_Profit"]


def profit_i(
    price_i,
    price_j,
    i,
    j,
    positions,
    coeff_dict,
    first_choice_data,
    book_titles,
    marginal_cost,
    price_other_books,
):
    """
    Computes the negative profit for book i (for minimization).
    """
    prices_candidate = [float(price_other_books)] * 10  # Ensure it's a list of floats
    prices_candidate[i] = float(price_i)  # Convert to float
    prices_candidate[j] = float(price_j)

    # print(f"prices_candidate in profit_i: {prices_candidate}")
    # print(f"prices_candidate shape: {np.array(prices_candidate).shape}")

    shares = market_shares(
        prices_candidate, positions, coeff_dict, first_choice_data, book_titles
    )
    share_i = next(share for book_idx, _, share in shares if book_idx == i)

    return -(price_i - marginal_cost) * share_i  # Negative for minimization


def profit_j(
    price_j,
    price_i,
    i,
    j,
    positions,
    coeff_dict,
    first_choice_data,
    book_titles,
    marginal_cost,
    price_other_books,
):
    """
    Computes the negative profit for book j (for minimization).
    """
    prices_candidate = [float(price_other_books)] * 10
    prices_candidate[i] = float(price_i)
    prices_candidate[j] = float(price_j)

    # print(f"prices_candidate in profit_j: {prices_candidate}")
    # print(f"prices_candidate shape: {np.array(prices_candidate).shape}")

    shares = market_shares(
        prices_candidate, positions, coeff_dict, first_choice_data, book_titles
    )
    share_j = next(share for book_idx, _, share in shares if book_idx == j)

    return -(price_j - marginal_cost) * share_j  # Negative for minimization


def find_nash_equilibrium(
    price_grid,
    i,
    j,
    positions,
    coeff_dict,
    first_choice_data,
    book_titles,
    marginal_cost,
    price_other_books,
    tolerance=0.01,
    max_iter=1000,
):
    """
    Finds the Nash equilibrium by iterating best responses using optimization.
    """
    # Initialize prices randomly
    price_i, price_j = np.mean(price_grid), np.mean(price_grid)
    converged = False
    iteration = 0

    while not converged and iteration < max_iter:
        prev_price_i, prev_price_j = price_i, price_j

        # Find best response for book i given current price_j
        res_i = minimize(
            profit_i,
            x0=[price_i],
            args=(
                price_j,
                i,
                j,
                positions,
                coeff_dict,
                first_choice_data,
                book_titles,
                marginal_cost,
                price_other_books,
            ),
            bounds=[(min(price_grid), max(price_grid))],
            method="L-BFGS-B",
        )
        price_i = res_i.x[0]

        # Find best response for book j given updated price_i
        res_j = minimize(
            profit_j,
            x0=[price_j],
            args=(
                price_i,
                i,
                j,
                positions,
                coeff_dict,
                first_choice_data,
                book_titles,
                marginal_cost,
                price_other_books,
            ),
            bounds=[(min(price_grid), max(price_grid))],
            method="L-BFGS-B",
        )
        price_j = res_j.x[0]

        # Check for convergence
        if (
            abs(price_i - prev_price_i) < tolerance
            and abs(price_j - prev_price_j) < tolerance
        ):
            converged = True

        iteration += 1

    # Compute final market shares and profits
    final_prices = [price_other_books] * 10
    final_prices[i] = price_i
    final_prices[j] = price_j

    shares = market_shares(
        final_prices, positions, coeff_dict, first_choice_data, book_titles
    )
    share_i = next(share for book_idx, _, share in shares if book_idx == i)
    share_j = next(share for book_idx, _, share in shares if book_idx == j)

    profit_i_val = (price_i - marginal_cost) * share_i
    profit_j_val = (price_j - marginal_cost) * share_j
    total_profit = profit_i_val + profit_j_val

    return price_i, price_j, profit_i_val, profit_j_val, total_profit


# def find_nash_equilibrium(price_grid, i, j, positions, coeff_dict, first_choice_data, book_titles, marginal_cost, price_other_books, tolerance=1e-1):
#     """
#     Finds the Nash equilibrium where both books maximize their individual profits.
#     """
#     # Initialize prices arbitrarily
#     price_i, price_j = np.random.choice(price_grid), np.random.choice(price_grid)
#     converged = False

#     while not converged:
#         prev_price_i, prev_price_j = price_i, price_j

#         # Optimize price for book i given book j's price
#         best_profit_i = -np.inf
#         for candidate_price_i in price_grid:
#             prices_candidate = [price_other_books] * 10
#             prices_candidate[i] = candidate_price_i
#             prices_candidate[j] = price_j

#             shares = market_shares(prices_candidate, positions, coeff_dict, first_choice_data, book_titles)
#             share_i = next(share for book_idx, _, share in shares if book_idx == i)
#             profit_i = (candidate_price_i - marginal_cost) * share_i

#             if profit_i > best_profit_i:
#                 best_profit_i = profit_i
#                 price_i = candidate_price_i

#         # Optimize price for book j given book i's price
#         best_profit_j = -np.inf
#         for candidate_price_j in price_grid:
#             prices_candidate = [price_other_books] * 10
#             prices_candidate[i] = price_i
#             prices_candidate[j] = candidate_price_j

#             shares = market_shares(prices_candidate, positions, coeff_dict, first_choice_data, book_titles)
#             share_j = next(share for book_idx, _, share in shares if book_idx == j)
#             profit_j = (candidate_price_j - marginal_cost) * share_j

#             if profit_j > best_profit_j:
#                 best_profit_j = profit_j
#                 price_j = candidate_price_j

#         # Check for convergence
#         if abs(price_i - prev_price_i) < tolerance and abs(price_j - prev_price_j) < tolerance:
#             converged = True

#     # Compute final market shares and profits
#     final_prices = [price_other_books] * 10
#     final_prices[i] = price_i
#     final_prices[j] = price_j
#     shares = market_shares(final_prices, positions, coeff_dict, first_choice_data, book_titles)
#     share_i = next(share for book_idx, _, share in shares if book_idx == i)
#     share_j = next(share for book_idx, _, share in shares if book_idx == j)
#     profit_i = (price_i - marginal_cost) * share_i
#     profit_j = (price_j - marginal_cost) * share_j
#     total_profit = profit_i + profit_j

#     return price_i, price_j, profit_i, profit_j, total_profit


def main():

    # ------------------------------
    # Load data and model coefficients
    # ------------------------------

    input_path = "data/experiment/input"
    first_choice_data, _, _, book_titles = load_data_long(input_path)

    date = "2025-03-21"
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

    # Define short names (helpful to shorten column names)
    model_short_names = {
        "observables-plain logit": "plain_logit",
        "observables-pages and year": "mlogit_attributes",
        "user_reviews_USE_text-PC1 and PC2": "mlogit_texts",
    }

    # Load coefficient dictionaries for each model
    model_coeff_dicts = {}
    for model_str in models_to_use:
        model_coeff_dicts[model_str] = load_coeff_dicts(all_sheets, model_str)

    # # ------------------------------
    # # Market shares example
    # # ------------------------------

    # # Example prices and positions
    # prices = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    # positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # # Compute market shares for each model
    # for model_str, coeff_dict in model_coeff_dicts.items():
    #     print(f"Computing market shares for model: {model_str}")

    #     # Example usage of market_shares function
    #     shares = market_shares(
    #         prices, positions, coeff_dict, first_choice_data, book_titles
    #     )

    #     for book_index, title, share in shares:
    #         print(f"Book {book_index} ({title}): {share:.4f}")
    #     print("\n")

    # ------------------------------
    # Merger simulations
    # ------------------------------

    # Reduce number of simulations to speed up computation
    M = 100
    first_choice_data = first_choice_data[
        first_choice_data["respondent_id"] <= M
    ].copy()
    # print(first_choice_data)

    # Load model estimates
    # model_str = "observables-plain logit"
    # model_str = "observables-pages and year"
    # model_str = "user_reviews_USE_text-PC1 and PC2"
    # coeff_dict = load_coeff_dicts(all_sheets, model_str)
    # print(coeff_dict)

    # Define grid of prices from $0 to $10 with step $1
    price_grid = np.arange(1, 200, 1)  # G elements

    # Fix free parameters
    marginal_cost = 0
    price_other_books = 5

    # Set initial positions
    positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Initialize an empty list to store results
    results = []

    # Keep book i fixed
    total_pairs = 9
    for i in range(10):

        # Possible other books to consider
        valid_j_values = [j for j in range(10) if j != i]

        # Loop over all book pairs
        for count, j in enumerate(valid_j_values, start=1):

            print()
            print(f"Processing book pair ({i+1}, {j+1}) - {count} out of {total_pairs}")

            # Initialize a dictionary to store results for this book pair
            row = {
                "book_i": i + 1,
                "book_j": j + 1,
                "title_i": book_titles[i],
                "title_j": book_titles[j],
            }

            # Compute market shares for each model
            for model_str, coeff_dict in model_coeff_dicts.items():
                print()
                print(f"Performing merger simulations for the model: {model_str}")

                # Find the price combination that maximizes total profit
                (
                    best_price_i,
                    best_price_j,
                    best_profit_i,
                    best_profit_j,
                    best_total_profit,
                ) = find_merged_optimum(
                    price_grid,
                    i,
                    j,
                    positions,
                    coeff_dict,
                    first_choice_data,
                    book_titles,
                    marginal_cost,
                    price_other_books,
                )

                # Print optimal price results
                print("\nOptimal Price Combination:")
                print(f"Price for Book {i+1}: ${best_price_i}")
                print(f"Price for Book {j+1}: ${best_price_j}")
                print(f"Profit for Book {i+1}: ${best_profit_i:.2f}")
                print(f"Profit for Book {j+1}: ${best_profit_j:.2f}")
                print(f"Total Profit: ${best_total_profit:.2f}")

                # Find Nash equilibrium prices
                (
                    nash_price_i,
                    nash_price_j,
                    nash_profit_i,
                    nash_profit_j,
                    nash_total_profit,
                ) = find_nash_equilibrium(
                    price_grid,
                    i,
                    j,
                    positions,
                    coeff_dict,
                    first_choice_data,
                    book_titles,
                    marginal_cost,
                    price_other_books,
                )

                # Report Nash equilibrium results
                print("\nNash Equilibrium Prices:")
                print(f"Price for Book {i+1}: ${nash_price_i}")
                print(f"Price for Book {j+1}: ${nash_price_j}")
                print(f"Profit for Book {i+1}: ${nash_profit_i:.2f}")
                print(f"Profit for Book {j+1}: ${nash_profit_j:.2f}")
                print(f"Total Profit: ${nash_total_profit:.2f}")

                # Compute absolute differences (before vs after merger)
                price_increase_i = best_price_i - nash_price_i
                price_increase_j = best_price_j - nash_price_j
                price_increase_avg = (price_increase_i + price_increase_j) / 2

                # Compute relative differences (before vs after merger)
                rel_price_increase_i = 100 * (best_price_i / nash_price_i - 1)
                rel_price_increase_j = 100 * (best_price_j / nash_price_j - 1)
                rel_price_increase_avg = (
                    rel_price_increase_i + rel_price_increase_j
                ) / 2

                # Display computed differences
                print("\nPrice Changes After Merger:")
                print(f"Price Increase for Book {i+1}: ${price_increase_i:.2f}")
                print(f"Price Increase for Book {j+1}: ${price_increase_j:.2f}")
                print(f"Average Price Increase: ${price_increase_avg:.2f}")
                print(f"Relative Average Price Increase: {rel_price_increase_avg:.2f}%")

                # Store results on predicted price increases
                short_model_str = model_short_names[model_str]
                # row[f"price_increase_i_{short_model_str}"] = price_increase_i
                # row[f"price_increase_j_{short_model_str}"] = price_increase_j
                row[f"price_increase_avg_{short_model_str}"] = price_increase_avg
                row[f"rel_price_increase_avg_{short_model_str}"] = (
                    rel_price_increase_avg
                )

            # Append the single row to the results list
            results.append(row)

    # Save to CSV
    df = pd.DataFrame(results)
    output_path = (
        "./data/experiment/output/merger_simulations/predicted_price_increases.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"Saved merger simulation results to {output_path}")


if __name__ == "__main__":
    main()
