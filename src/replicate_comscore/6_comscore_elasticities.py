# python -m src.replicate_comscore.6_comscore_elasticities

import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from src.helper_functions.estimation.estimate_mixed_logit import (
    generate_predicated_diversion_matrix,
)
from src.helper_functions.estimation.load_data import (
    load_comscore_categories,
    load_data_long_comscore,
)
from src.helper_functions.estimation.load_model_results import (
    generate_util_matrix,
    load_coeff_dicts,
)


# Define the process_draw function outside the main function
def process_draw(d, choice_data, model_coeff, draws):
    """
    Process a single draw for computing elasticities.

    Parameters:
    - d: Draw index.
    - choice_data: DataFrame containing the choice data.
    - model_coeff: Dictionary containing the model coefficients.
    - draws: Dictionary of random draws for each random coefficient.

    Returns:
    - s_ijd: Choice probabilities for this draw.
    - der_own_ijd: Derivatives for own-price elasticities.
    - der_cross_ijkd: Derivatives for cross-price elasticities.
    """
    # Initialize model coefficients for this draw
    model_coeff_d = model_coeff.copy()
    for std_coeff in draws.keys():
        mean_coeff = std_coeff[3:]
        model_coeff_d[mean_coeff] = (
            model_coeff[mean_coeff]
            + draws[std_coeff][d] * model_coeff[std_coeff]  # mean + std * draw
        )
    # Remove standard deviation keys from model_coeff_d so generate_util_matrix doesn't draw again
    model_coeff_d = {
        key: model_coeff_d[key] for key in model_coeff.keys() if key[:3] != "sd."
    }

    # Generate utility matrix for this draw (this is now deterministic)
    util_matrix = generate_util_matrix(
        choice_data, model_coeff_d, include_gumbels=False
    )  # Size (N, J)

    # Compute choice probabilities for this draw
    exp_util = np.exp(util_matrix)
    sum_exp_util = np.sum(exp_util, axis=1, keepdims=True)
    s_ijd = exp_util / sum_exp_util  # Size (N, J)

    # Compute derivatives for own-price elasticities
    der_own_ijd = s_ijd * (1 - s_ijd) * model_coeff_d["price"]

    N, J = s_ijd.shape

    # Compute derivatives for cross-price elasticities
    der_cross_ijkd = np.zeros((N, J, J))
    for j in range(J):
        for k in range(J):
            if j != k:
                der_cross_ijkd[:, j, k] = (
                    -s_ijd[:, j] * s_ijd[:, k] * model_coeff_d["price"]
                )

    return s_ijd, der_own_ijd, der_cross_ijkd


def compute_elasticities_matrix_parallel(choice_data, model_coeff, D=10000, seed=123):
    """
    Compute the elasticities matrix for a model given choice data and model coefficients.
    Parallelized using multiprocessing.

    Parameters:
    - choice_data: DataFrame containing the choice data.
    - model_coeff: Dictionary containing the model coefficients.
    - D: Number of draws for the simulation.
    - seed: Random seed for reproducibility.

    Returns:
    - elasticities_matrix: J x J matrix of elasticities.
    """
    np.random.seed(seed)

    choice_ids = choice_data["choice_id"].unique()
    unique_products = choice_data["product_id"].unique()
    N = len(choice_ids)
    J = len(unique_products)

    # Generate random draws for each random coefficient
    draws = {}
    for coeff in model_coeff.keys():
        if coeff[:3] == "sd.":
            model_coeff[coeff] = abs(model_coeff[coeff])  # SDs must be positive
            draws[coeff] = np.random.normal(0, 1, D)  # Standard normal draws

    # Use multiprocessing to parallelize the draws
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_draw, [(d, choice_data, model_coeff, draws) for d in range(D)]
        )

    # Combine results from all draws
    s_ij = np.zeros((N, J))
    der_own_ij = np.zeros((N, J))
    der_cross_ijk = np.zeros((N, J, J))

    for s_ijd, der_own_ijd, der_cross_ijkd in results:
        s_ij += s_ijd / D
        der_own_ij += der_own_ijd / D
        der_cross_ijk += der_cross_ijkd / D

    # Obtain prices from choice data
    price_ij = choice_data.pivot(
        index="choice_id", columns="product_id", values="price"
    ).values  # Size (N, J)

    # Compute own-price elasticities
    own_elasticities = np.mean(
        der_own_ij * price_ij / s_ij, axis=0
    )  # Size (J,), average over consumers i

    # Compute cross-price elasticities
    cross_elasticities = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j != k:
                cross_elasticities[j, k] = np.mean(  # average over consumers i
                    der_cross_ijk[:, j, k] * price_ij[:, k] / s_ij[:, j]
                )

    # Combine into a single J x J matrix
    elasticities_matrix = np.zeros((J, J))
    np.fill_diagonal(elasticities_matrix, own_elasticities)
    for j in range(J):
        for k in range(J):
            if j != k:
                elasticities_matrix[j, k] = cross_elasticities[j, k]

    return elasticities_matrix


def load_best_AIC_model(estimate_results_path, category, date):
    estimation_results_path = os.path.join(
        estimate_results_path, f"mixed_logit_results_{category}_{date}.xlsx"
    )
    try:
        all_sheets = pd.read_excel(estimation_results_path, sheet_name=None)
    except FileNotFoundError:
        # try with date + 1
        date = pd.to_datetime(date) + pd.DateOffset(days=1)
        date = date.strftime("%Y-%m-%d")
        estimation_results_path = os.path.join(
            estimate_results_path, f"mixed_logit_results_{category}_{date}.xlsx"
        )
        all_sheets = pd.read_excel(estimation_results_path, sheet_name=None)

    best_AIC = float("inf")
    best_AIC_model_str = None
    plain_logit_model_str = None
    for sheet_name, df_sheet in all_sheets.items():
        if sheet_name == "combined":
            continue
        row = df_sheet.loc[df_sheet["First Choice AIC"].idxmin()]
        if row["First Choice AIC"] < best_AIC:
            best_AIC = row["First Choice AIC"]
            best_AIC_model_str = f"{sheet_name}-{row['Specification']}"
            plain_logit_model_str = f"{sheet_name}-plain logit"

    print(f"Path: {estimation_results_path}")
    print(f"Best AIC model: {best_AIC_model_str}")
    model_coeff, model_se = load_coeff_dicts(
        all_sheets, best_AIC_model_str, include_se=True
    )

    plain_model_coeff, plain_model_se = load_coeff_dicts(
        all_sheets, plain_logit_model_str, include_se=True
    )

    return best_AIC_model_str, model_coeff, model_se, plain_model_coeff, plain_model_se


def check_sign_test(elasticities_matrix):
    if np.all(np.diag(elasticities_matrix) < 0) and np.all(
        elasticities_matrix[~np.eye(elasticities_matrix.shape[0], dtype=bool)] > 0
    ):
        return True
    else:
        return False


def compute_average_range_diversion_ratios(diversion_matrix):
    J = diversion_matrix.shape[1]
    average_range = 0
    for j in range(J):
        diversion_ratios = np.delete(diversion_matrix[j], j)
        assert np.isclose(np.sum(diversion_ratios), 1)
        average_range += np.ptp(diversion_ratios)

    return average_range / J


def compute_average_hhi_diversion_ratios(diversion_matrix):
    J = diversion_matrix.shape[1]
    average_hhi = 0
    for j in range(J):
        diversion_ratios = np.delete(diversion_matrix[j], j)
        assert np.isclose(np.sum(diversion_ratios), 1)
        average_hhi += np.sum(diversion_ratios**2)

    return average_hhi / J


def compute_max_discrepancy(elasticities_matrix):
    # Positive own-elasticities
    positive_own_elasticities = np.diagonal(elasticities_matrix)
    positive_own_elasticities = positive_own_elasticities[positive_own_elasticities > 0]
    max_positive_own = (
        np.max(positive_own_elasticities) if positive_own_elasticities.size > 0 else 0
    )

    # Negative cross-elasticities
    negative_cross_elasticities = np.delete(
        elasticities_matrix, np.arange(elasticities_matrix.shape[0]), axis=1
    )
    negative_cross_elasticities = negative_cross_elasticities[
        negative_cross_elasticities < 0
    ]
    min_negative_cross = (
        np.min(negative_cross_elasticities)
        if negative_cross_elasticities.size > 0
        else 0
    )

    # Max discrepancy
    max_discrepancy = max(abs(max_positive_own), abs(min_negative_cross))

    return max_discrepancy


def save_matrix_to_csv(matrix, choice_data, output_path, category, file_name):
    output_path = os.path.join(output_path, category)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.DataFrame(matrix)
    df.index = choice_data["product_id"].unique()
    df.columns = choice_data["product_id"].unique()

    file_name = f"{file_name}_{category}.csv"
    df.to_csv(os.path.join(output_path, file_name), index=True)


def main():

    date = "2025-03-19"
    input_path = "data/comscore/intermediate"
    estimate_results_path = "data/comscore/output/estimation_results/"
    output_path = "data/comscore/output/elasticities"
    D = 10000  # Switch to 10,000 for final run
    save_elasticities_df = True
    save_diversion_df = True

    categories = load_comscore_categories()

    elasticities_summary = {}

    for category in categories:
        print(f"\n\nCategory: {category}\n")

        # 1. Load choice data
        choice_data_path = os.path.join(input_path, category)
        choice_data = load_data_long_comscore(choice_data_path)

        # 2. Load model coefficients
        (
            model_str,
            model_coeff,
            model_se,
            plain_logit_model_coeff,
            plain_logit_model_se,
        ) = load_best_AIC_model(estimate_results_path, category, date)
        assert model_str is not None

        # 3. Compute elasticities matrix
        elasticities_matrix = compute_elasticities_matrix_parallel(
            choice_data, model_coeff, D=D
        )
        if save_elasticities_df:
            save_matrix_to_csv(
                elasticities_matrix, choice_data, output_path, category, "elasticities"
            )

        # 5. Compute diversion matrix
        pca_diversion_matrix = generate_predicated_diversion_matrix(
            model_coeff, choice_data
        )
        plain_logit_diversion_matrix = generate_predicated_diversion_matrix(
            plain_logit_model_coeff, choice_data
        )
        if save_diversion_df:
            save_matrix_to_csv(
                pca_diversion_matrix,
                choice_data,
                output_path,
                category,
                "pca_diversion",
            )
            save_matrix_to_csv(
                plain_logit_diversion_matrix,
                choice_data,
                output_path,
                category,
                "plain_logit_diversion",
            )

        elasticities_summary[category] = {
            "Model": model_str.split("-")[0],
            "Specifications": model_str.split("-")[1],
            "PCA Price Estimate": model_coeff.get("price", "n/a"),
            "PCA Price SE": model_se.get("price", "n/a"),
            "PCA Price Std Estimate": model_coeff.get("sd.price", "n/a"),
            "PCA Price Std SE": model_se.get("sd.price", "n/a"),
            "Plain Logit Price Estimate": plain_logit_model_coeff.get("price", "n/a"),
            "Plain Logit Price SE": plain_logit_model_se.get("price", "n/a"),
            "Average range of elasticity columns": np.mean(
                np.ptp(elasticities_matrix, axis=0)
            ),
            "Pass Sign Test?": check_sign_test(elasticities_matrix),
            "Max discrepancy of elasticity matrix": compute_max_discrepancy(
                elasticities_matrix
            ),
            "N (consumers)": len(choice_data["choice_id"].unique()),
            "J (products)": len(choice_data["product_id"].unique()),
            "PCA Average range of diversion ratios": compute_average_range_diversion_ratios(
                pca_diversion_matrix
            ),
            "Plain Logit Average range of diversion ratios": compute_average_range_diversion_ratios(
                plain_logit_diversion_matrix
            ),
            "PCA Average HHI of diversion ratios": compute_average_hhi_diversion_ratios(
                pca_diversion_matrix
            ),
            "Plain Logit Average HHI of diversion ratios": compute_average_hhi_diversion_ratios(
                plain_logit_diversion_matrix
            ),
        }
        elasticities_summary_df = pd.DataFrame(elasticities_summary).T
        elasticities_summary_df.to_csv(
            os.path.join(output_path, "elasticity_summary.csv"), index=True
        )
        print(
            f"Saved progress to {os.path.join(output_path, 'elasticity_summary_in_progress.csv')}"
        )

    elasticities_summary_df = pd.DataFrame(elasticities_summary).T
    elasticities_summary_df.to_csv(
        os.path.join(output_path, "elasticity_summary.csv"), index=True
    )


if __name__ == "__main__":
    main()
