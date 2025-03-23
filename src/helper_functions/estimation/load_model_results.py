import numpy as np

from src.helper_functions.estimation.estimate_mixed_logit import simulate_individual


def parse_coefficients_as_dict(coeff_names_str, coeff_values_str):
    """
    Given two strings that look like:
        coeff_names_str  -> "['product_id_3' 'product_id_8' 'price' ...]"
        coeff_values_str -> "[ 0.55129951 -0.49843977 -0.01164972 ...]"
    parse them into a Python dictionary: {name: value, ...}.
    """
    # Remove the '[' and ']' and split
    # Then split on whitespace
    names_clean = coeff_names_str.strip("[]").split()
    coeff_names = [token.strip("'") for token in names_clean]
    values_clean = coeff_values_str.strip("[]").split()
    coeff_values = [float(x) for x in values_clean]

    coeff_dict = dict(zip(coeff_names, coeff_values))
    return coeff_dict


def load_coeff_dicts(all_sheets, model_str, include_se=False):
    sheet_part, spec_part = model_str.split("-", 1)
    sheet_part = sheet_part.strip()
    spec_part = spec_part.strip()

    if sheet_part not in all_sheets:
        raise ValueError(f"Sheet '{sheet_part}' not in Excel file.")

    df_sheet = all_sheets[sheet_part]
    row = df_sheet.loc[df_sheet["Specification"] == spec_part]
    if row.empty:
        raise ValueError(
            f"Specification '{spec_part}' not found in sheet '{sheet_part}'."
        )

    coeff_names_str = row["Coefficient Names"].values[0]
    coeff_values_str = row["Estimated Coefficients"].values[0]

    if isinstance(coeff_names_str, str):
        coeff_dict = parse_coefficients_as_dict(coeff_names_str, coeff_values_str)
    else:
        coeff_dict = dict(zip(coeff_names_str, coeff_values_str))

    if include_se:
        se_str = row["Standard Errors"].values[0]
        # se_str = " ".join(["0.1"] * len(coeff_dict))
        if isinstance(coeff_names_str, str):
            se_dict = parse_coefficients_as_dict(coeff_names_str, se_str)
        else:
            se_dict = dict(zip(coeff_names_str, se_str))

        return coeff_dict, se_dict

    return coeff_dict


def generate_util_matrix(
    first_choice_data,
    coeff_dict,
    seed_offset=123,
    empirical_diversion_matrix=None,
    include_gumbels=True,
):
    choice_ids = first_choice_data["choice_id"].unique()
    unique_products = first_choice_data["product_id"].unique()
    N = len(choice_ids)
    J = len(unique_products)

    util_matrix = np.zeros((N, J))
    predicted_count_matrix = np.zeros((J, J), dtype=float)

    for i, cid in enumerate(choice_ids):
        utilities = simulate_individual(
            first_choice_data,
            coeff_dict,
            cid,
            seed=seed_offset + cid,
            include_gumbels=include_gumbels,
        )
        util_matrix[i, :] = utilities

        if empirical_diversion_matrix is not None:
            sorted_idx = np.argsort(utilities)[::-1]
            first_choice_idx = sorted_idx[0]
            second_choice_idx = sorted_idx[1]

            first_choice_product_id = unique_products[first_choice_idx]
            second_choice_product_id = unique_products[second_choice_idx]

            row_index_for_first = np.where(unique_products == first_choice_product_id)[
                0
            ][0]
            col_index_for_second = np.where(
                unique_products == second_choice_product_id
            )[0][0]
            predicted_count_matrix[row_index_for_first, col_index_for_second] += 1

    if empirical_diversion_matrix is not None:
        col_sums = predicted_count_matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            predicted_diversion_matrix = np.where(
                col_sums != 0, predicted_count_matrix / col_sums, 0
            )

        assert np.allclose(predicted_diversion_matrix.sum(axis=1), 1)
        assert np.allclose(empirical_diversion_matrix.sum(axis=1), 1)

        rmse = np.sqrt(
            np.mean((empirical_diversion_matrix - predicted_diversion_matrix) ** 2)
        )

        return util_matrix, rmse

    return util_matrix
