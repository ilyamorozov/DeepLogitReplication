# python -m src.replicate_comscore.6_comscore_elasticities

import os
from multiprocessing import Pool, cpu_count
from copy import deepcopy
import numpy as np
import pandas as pd

from src.helper_functions.estimation.estimate_mixed_logit import (
    estimate_mixed_logit, 
    predict_mixed_logit,
    generate_predicted_diversion_matrix,
)
from src.helper_functions.estimation.load_data import (
    load_comscore_categories,
    load_data_long_comscore,
)
from src.helper_functions.estimation.load_model_results import (
    generate_util_matrix,
    load_coeff_dicts,
)
from src.helper_functions.file_structure.get_file_path_from_config import get_file_path_from_config

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


def load_best_AIC_model(estimate_results_path, category, date, main_categories=None):
    if main_categories is None:
        main_categories = {"13060404", "13060114", "13060701", "13110101"}

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

    # Find 'observables' sheet in a case-insensitive way (if it exists)
    observables_sheet_name = next((s for s in all_sheets.keys() if s.lower() == "observables"), None)

    # Best AIC across all sheets except 'combined' and 'observables'
    best_AIC = float("inf")
    best_sheet_for_plain = None
    best_AIC_model_str = None

    for sheet_name, df_sheet in all_sheets.items():
        if sheet_name.lower() in {"combined", "observables"}:
            continue
        if "First Choice AIC" not in df_sheet.columns or df_sheet.empty:
            continue
        row = df_sheet.loc[df_sheet["First Choice AIC"].idxmin()]
        if pd.notna(row["First Choice AIC"]) and row["First Choice AIC"] < best_AIC:
            best_AIC = row["First Choice AIC"]
            best_AIC_model_str = f"{sheet_name}-{row['Specification']}"
            best_sheet_for_plain = sheet_name

    if best_sheet_for_plain is None:
        raise ValueError("Could not identify best sheet for overall best-AIC model.")
    plain_logit_model_str = f"{best_sheet_for_plain}-plain logit"

    plain_logit_AIC = float(
        all_sheets[best_sheet_for_plain].loc[
            all_sheets[best_sheet_for_plain]["Specification"] == "plain logit",
            "First Choice AIC"
        ].values[0]
    )

    # Attr-based best within 'observables' (only for main categories)
    attr_model_str = None
    if category in main_categories and observables_sheet_name is not None:
        df_obs = all_sheets[observables_sheet_name]
        if "First Choice AIC" in df_obs.columns and not df_obs.empty:
            row_obs = df_obs.loc[df_obs["First Choice AIC"].idxmin()]
            if pd.notna(row_obs["First Choice AIC"]):
                attr_model_str = f"{observables_sheet_name}-{row_obs['Specification']}"
                attr_best_AIC = row_obs["First Choice AIC"]

    print(f"Category: {category}, Date: {date}")
    print(f"Best PCA model: {best_AIC_model_str} with AIC {best_AIC}")
    print(f"Plain Logit model: {plain_logit_model_str} with AIC {plain_logit_AIC}")
    if attr_model_str:
        print(f"Attr-based: {attr_model_str} with AIC {attr_best_AIC}")

    # Load coefficients
    model_coeff, model_se = load_coeff_dicts(all_sheets, best_AIC_model_str, include_se=True)
    plain_model_coeff, plain_model_se = load_coeff_dicts(all_sheets, plain_logit_model_str, include_se=True)

    if attr_model_str:
        attr_model_coeff, attr_model_se = load_coeff_dicts(all_sheets, attr_model_str, include_se=True)
    else:
        attr_best_AIC, attr_model_coeff, attr_model_se = None, None, None

    return (
        best_AIC_model_str,
        best_AIC, 
        model_coeff,
        model_se,
        plain_logit_AIC,
        plain_model_coeff,
        plain_model_se,
        attr_model_str,
        attr_best_AIC,
        attr_model_coeff,
        attr_model_se,
    )


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


def save_matrix_to_csv(matrix, choice_data, output_path, category, file_name, attr = False):
    output_path = os.path.join(output_path, category)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.DataFrame(matrix)
    df.index = choice_data["product_id"].unique()
    df.columns = choice_data["product_id"].unique()

    file_name = f"{file_name}_{category}.csv"
    if attr: 
        file_name = f"attr_based_{file_name}"
    df.to_csv(os.path.join(output_path, file_name), index=True)

def _aligned_vec_from_dict(names, coeff_dict, fill_current=None):
    """Build a vector aligned to `names` from `coeff_dict`, optionally
    falling back to `fill_current[i]` when the name is missing in dict."""
    vec = np.zeros(len(names), dtype=float)
    for i, n in enumerate(names):
        if n in coeff_dict:
            vec[i] = float(coeff_dict[n])
        else:
            vec[i] = float(fill_current[i]) if fill_current is not None else 0.0
    return vec

def _overwrite_if_diff(model, coeff_dict, label="", rtol=1e-6, atol=1e-8):
    """If model.coeff_ differs from the Excel dict, overwrite model.coeff_ and print a message."""
    names = list(model.coeff_names)
    cur = np.array(model.coeff_, dtype=float)
    new = _aligned_vec_from_dict(names, coeff_dict, fill_current=cur)

    if not np.allclose(cur, new, rtol=rtol, atol=atol):
        # stats for message
        diffs = np.abs(cur - new)
        num_changed = int(np.sum(diffs > np.maximum(atol, np.abs(new) * rtol)))
        max_abs = float(np.max(diffs))

        # overwrite core vector
        model.coeff_ = new

        # if there is a mapping from name->index, keep it consistent 
        if hasattr(model, "coeff_names"):
            model.coeff_names = names  # unchanged order, but explicit

        print(
            "[info] Overwrote coefficients from Excel "
            + (f"for {label} " if label else "")
            + f"({num_changed}/{len(names)} entries differed; max|Δ|={max_abs:.3g})."
        )

def main():

    date = "2025-08-21"

    input_path = get_file_path_from_config(path_type="COMSCORE_5", path="INPUT_PATH")
    intermediate_path = get_file_path_from_config(path_type="COMSCORE_5", path="INTERMEDIATE_PATH")
    estimate_results_path = get_file_path_from_config(path_type="COMSCORE_5", path="RESULT_PATH")
    output_path = get_file_path_from_config(path_type="COMSCORE_5", path="OUTPUT_PATH")

    D = 10000  # Switch to 10,000 for final run
    save_elasticities_df = True
    save_diversion_df = True

    categories = load_comscore_categories(input_path)
    diversions_summary = {}

    for category in categories:
        print(f"\n\nCategory: {category}\n")

        # 1. Load choice data
        choice_data_path = os.path.join(intermediate_path, category)
        choice_data = load_data_long_comscore(choice_data_path)

        # 2. Load model coefficients
        (
            model_str,
            model_AIC,
            model_coeff,
            model_se,
            plain_logit_AIC,
            plain_logit_model_coeff,
            plain_logit_model_se,
            attr_model_str,
            attr_model_AIC,
            attr_model_coeff,
            attr_model_se,
        ) = load_best_AIC_model(estimate_results_path, category, date)
        assert model_str is not None

        
        product_fixed_effects_varnames_minus_1 = sorted(
            [
                col
                for col in choice_data.columns
                if col.startswith("product_id_")
            ]
        )[:-1]
        varnames = product_fixed_effects_varnames_minus_1 + ["price"]

        # re-estimate the model
        plain_logit_model = estimate_mixed_logit(
            choice_data,
            varnames=varnames,
            randvars={},  # plain logit has no random coeffs
            n_draws=100,
            num_starting_points=100,
            seed=1,
            halton=False,
        )
        s_unconditional = predict_mixed_logit(plain_logit_model, choice_data, varnames)
        plain_logit_diversion_matrix = generate_predicted_diversion_matrix(
            choice_data,
            s_unconditional,
            plain_logit_model,
            varnames,
        )

        if not np.isclose(plain_logit_model.aic, plain_logit_AIC, rtol=1e-6, atol=1e-8):
            print(f"Overwrote plain logit AIC from Excel ({plain_logit_AIC} -> {plain_logit_model.aic})")
            plain_logit_model.aic = plain_logit_AIC

        _overwrite_if_diff(plain_logit_model, plain_logit_model_coeff, label="plain_logit")

        save_matrix_to_csv(
            plain_logit_diversion_matrix,
            choice_data,
            output_path,
            category,
            "plain_logit_diversion",
        )
        embedding_model = model_str.split("-")[0]
        num_PCs = sum(col.startswith(f"{embedding_model}_pc") for col in choice_data.columns)
        k = min(6, num_PCs)

        pc_varnames = (
            product_fixed_effects_varnames_minus_1
            + ["price"]
            + [f"{embedding_model}_pc{i}" for i in range(1, k+1)]
        )

        sd_var = {k: v for k, v in model_coeff.items() if k.startswith("sd.")}
        sd_var = {k[3:]: v for k, v in sd_var.items()}
        randvars = {var: "n" for var in sd_var}

        # Validate that all randvars are in pc_varnames
        missing = set(randvars.keys()) - set(pc_varnames)
        if missing:
            print(f"[WARNING] randvars keys not in pc_varnames: {missing}")
            print(f"  randvars keys: {list(randvars.keys())}")
            print(f"  pc_varnames (last {k+1}): {pc_varnames[-(k+1):]}")
            # Filter to only valid randvars
            randvars = {var: "n" for var in randvars if var in pc_varnames}
            print(f"  Filtered randvars: {list(randvars.keys())}")

        best_model = estimate_mixed_logit(
            choice_data,
            varnames=pc_varnames,
            randvars=randvars,
            n_draws=100,
            num_starting_points=100,
            seed=1,
            halton=False,
        )

        if not np.isclose(best_model.aic, model_AIC, rtol=1e-6, atol=1e-8):
            print(f"Overwrote plain logit AIC from Excel ({model_AIC} -> {best_model.aic})")
            best_model.aic = model_AIC

        _overwrite_if_diff(best_model, model_coeff, label=f"{model_str}")

        s_unconditional = predict_mixed_logit(best_model, choice_data, pc_varnames)
        pca_diversion_matrix = generate_predicted_diversion_matrix(
            choice_data,
            s_unconditional,
            best_model,
            pc_varnames,
        )

        save_matrix_to_csv(
            pca_diversion_matrix,
            choice_data,
            output_path,
            category,
            "pca_diversion",
        )
        if attr_model_str is not None:
            embedding_model = attr_model_str.split("-")[0]

            attr_varnames = (
                product_fixed_effects_varnames_minus_1
                + ["price"]
                + [f"{embedding_model}_pc{i}" for i in range(1, k+1)]
            )

            sd_var_attr = {k: v for k, v in attr_model_coeff.items() if k.startswith("sd.")}
            sd_var_attr = {k[3:]: v for k, v in sd_var_attr.items()}
            randvars_attr = {var: "n" for var in sd_var_attr}

            # Validate that all randvars are in attr_varnames
            missing_attr = set(randvars_attr.keys()) - set(attr_varnames)
            if missing_attr:
                print(f"[WARNING] attr randvars keys not in attr_varnames: {missing_attr}")
                randvars_attr = {var: "n" for var in randvars_attr if var in attr_varnames}

            attr_model = estimate_mixed_logit(
                choice_data,
                varnames=attr_varnames,
                randvars=randvars_attr,
                n_draws=100,
                num_starting_points=100,
                seed=1,
                halton=False,
            )

            if not np.isclose(attr_model.aic, attr_model_AIC, rtol=1e-6, atol=1e-8):
                print(f"Overwrote attr based AIC from Excel ({attr_model_AIC} -> {attr_model.aic})")
                attr_model.aic = attr_model_AIC
            _overwrite_if_diff(attr_model, attr_model_coeff, label=f"{attr_model_str}")

            s_unconditional = predict_mixed_logit(attr_model, choice_data, attr_varnames)
            attr_diversion_matrix = generate_predicted_diversion_matrix(
                choice_data,
                s_unconditional,
                attr_model,
                attr_varnames,
            )

            save_matrix_to_csv(
                attr_diversion_matrix,
                choice_data,
                output_path,
                category,
                "pca_diversion",
                attr=True
            )

        # # 3. Compute elasticities matrix
        elasticities_matrix = compute_elasticities_matrix_parallel(
            choice_data, model_coeff, D=D
        )
        if save_elasticities_df:
            save_matrix_to_csv(
                elasticities_matrix, choice_data, output_path, category, "elasticities"
            )

        diversions_summary[category] = {
            "Model": model_str.split("-")[0],
            "Specifications": model_str.split("-")[1],
            "PCA Price Estimate": model_coeff.get("price", "n/a"),
            "PCA Price SE": model_se.get("price", "n/a"),
            "PCA Price Std Estimate": model_coeff.get("sd.price", "n/a"),
            "PCA Price Std SE": model_se.get("sd.price", "n/a"),
            "Plain Logit Price Estimate": plain_logit_model_coeff.get("price", "n/a"),
            "Plain Logit Price SE": plain_logit_model_se.get("price", "n/a"),
    #         # "Average range of elasticity columns": np.mean(
    #         #     np.ptp(elasticities_matrix, axis=0)
    #         # ),
    #         # "Pass Sign Test?": check_sign_test(elasticities_matrix),
    #         # "Max discrepancy of elasticity matrix": compute_max_discrepancy(
    #         #     elasticities_matrix
    #         # ),
    #         "N (consumers)": len(choice_data["choice_id"].unique()),
    #         "J (products)": len(choice_data["product_id"].unique()),
    #         "PCA Average range of diversion ratios": compute_average_range_diversion_ratios(
    #             pca_diversion_matrix
    #         ),
    #         "Plain Logit Average range of diversion ratios": compute_average_range_diversion_ratios(
    #             plain_logit_diversion_matrix
    #         ),
    #         "PCA Average HHI of diversion ratios": compute_average_hhi_diversion_ratios(
    #             pca_diversion_matrix
    #         ),
    #         "Plain Logit Average HHI of diversion ratios": compute_average_hhi_diversion_ratios(
    #             plain_logit_diversion_matrix
    #         ),
            "PCA Average of diversion ratios to closest substitute": 
                pca_diversion_matrix.max(axis=1).mean(),

            "Plain Logit Average of diversion ratios to closest substitute":
                plain_logit_diversion_matrix.max(axis=1).mean(),

            "Attribute-Based PCA Average of diversion ratios to closest substitute":
                attr_diversion_matrix.max(axis=1).mean() if attr_model_str is not None else np.nan,

            "PCA AIC": best_model.aic,

            "Plain Logit AIC": plain_logit_model.aic,

            "Attribute-Based PCA AIC": attr_model.aic if attr_model_str is not None else np.nan,
        }

        diversions_summary_df = pd.DataFrame(diversions_summary).T
        diversions_summary_df.to_csv(
            os.path.join(output_path, "diversions_summary.csv"), index=True
        )
        print(
            f"Saved progress to {os.path.join(output_path, 'diversions_summary_in_progress.csv')}"
        )

    diversions_summary_df = pd.DataFrame(diversions_summary).T
    diversions_summary_df.to_csv(
        os.path.join(output_path, "diversions_summary.csv"), index=True
    )
    print("All categories processed.")

    #     elasticities_summary_df = pd.DataFrame(elasticities_summary).T
    #     elasticities_summary_df.to_csv(
    #         os.path.join(output_path, "elasticities_summary.csv"), index=True
    #     )
    #     print(
    #         f"Saved progress to {os.path.join(output_path, 'elasticity_summary_in_progress.csv')}"
    #     )

    # elasticities_summary_df = pd.DataFrame(elasticities_summary).T
    # elasticities_summary_df.to_csv(
    #     os.path.join(output_path, "elasticities_summary.csv"), index=True
    # )

if __name__ == "__main__":
    main()


