# python -m src.replicate_experiment.3_experiment_estimation_mixed_logit

import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.helper_functions.estimation.estimate_mixed_logit import (
    compute_second_choice_likelihood,
    compute_second_choice_rmse,
    estimate_mixed_logit,
    load_data_long,
)


def save_results_to_xlsx(results, output_path):
    """
    Save results dictionary to an XLSX file where each key becomes a separate sheet.

    Args:
        results (dict): Dictionary where each key is a model name and value is a dict of specifications
        output_path (str): Path where the XLSX file should be saved
    """
    # Create ExcelWriter object
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Iterate through each model in results
        for model_name, specifications in results.items():
            # Convert the specifications dictionary to a DataFrame
            df = pd.DataFrame.from_dict(specifications, orient="index")

            # Reset index to make specification names a column
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Specification"}, inplace=True)

            # Reorder columns to match desired format
            columns = [
                "Specification",
                "First Choice LL",
                "First Choice AIC",
                "Second Choice LL",
                "Second Choice RMSE",
                "Coefficient Names",
                "Estimated Coefficients",
                "Likelihood Ratio Test",
            ]
            df = df[columns]

            # Write to Excel
            df.to_excel(
                writer,
                sheet_name=model_name,
                index=False,
            )

            # Get the worksheet to apply formatting
            worksheet = writer.sheets[model_name]

            # Format header
            for cell in worksheet[1]:
                cell.style = "Headline 2"


def save_predicted_diversion_matrices_to_xlsx(
    results, output_path, empirical_diversion_matrix, book_titles
):
    """
    Create a multi-sheet Excel file where each sheet contains the predicted diversion
    matrix (10 x 10) for the best model specification (lowest AIC) under each top-level
    key in 'results'. Also includes sheets for the empirical diversion matrix
    and the plain logit diversion matrix.
    """
    # Make sure the directory for output_path already exists before writing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(results.keys())

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 1. Empirical Diversion Matrix
        df_empirical = pd.DataFrame(
            empirical_diversion_matrix, index=book_titles, columns=book_titles
        )
        df_empirical.to_excel(writer, sheet_name="data")

        # 2. Plain Logit Diversion Matrix (assumes 'plain logit' under 'observables')
        if "observables" in results and "plain logit" in results["observables"]:
            plain_logit_matrix = results["observables"]["plain logit"].get(
                "Predicted Diversion Matrix"
            )
            if plain_logit_matrix is not None:
                df_plain = pd.DataFrame(
                    plain_logit_matrix, index=book_titles, columns=book_titles
                )
                df_plain.to_excel(writer, sheet_name="plain_logit")

        # 3. For each top-level key in 'results', find the best specification by AIC
        for top_key, specs_dict in results.items():

            best_spec = None
            best_aic = float("inf")

            for spec_name, spec_data in specs_dict.items():
                aic_value = spec_data.get("First Choice AIC", float("inf"))
                if aic_value <= best_aic:
                    best_aic = aic_value
                    best_spec = spec_name

            # If we found a best spec, retrieve & save its diversion matrix
            if best_spec is not None:
                best_pred_matrix = specs_dict[best_spec].get(
                    "Predicted Diversion Matrix"
                )
                if best_pred_matrix is not None:
                    df_pred = pd.DataFrame(
                        best_pred_matrix, index=book_titles, columns=book_titles
                    )
                    sheet_name = f"{str(top_key)}"
                    df_pred.to_excel(writer, sheet_name=sheet_name)


def save_model_to_results(
    model,
    subresults,
    specification,
    first_choice_data,
    second_choice_data,
    varnames,
    empirical_diversion_matrix,
):

    if subresults == {}:
        second_choice_ll = compute_second_choice_likelihood(
            model,
            first_choice_data,
            second_choice_data,
            varnames,
        )
        rmse, predicted_diversion_matrix = compute_second_choice_rmse(
            model,
            first_choice_data,
            empirical_diversion_matrix,
            return_predicted_diversion_matrix=True,
        )

    else:
        best_model_subset = "plain logit"
        for model_subset in subresults:
            if (
                set(subresults[model_subset]["Coefficient Names"]).issubset(
                    set(model.coeff_names)
                )
                and subresults[model_subset]["First Choice LL"]
                > subresults[best_model_subset]["First Choice LL"]
            ):
                best_model_subset = model_subset

        if model.loglikelihood < subresults[best_model_subset]["First Choice LL"]:
            model.loglikelihood = subresults[best_model_subset]["First Choice LL"]
            model.aic = subresults[best_model_subset]["First Choice AIC"]
            second_choice_ll = subresults[best_model_subset]["Second Choice LL"]
            rmse = subresults[best_model_subset]["Second Choice RMSE"]
            predicted_diversion_matrix = subresults[best_model_subset][
                "Predicted Diversion Matrix"
            ]
            model.coeff_ = np.zeros(len(model.coeff_names))
            for i in range(len(subresults[best_model_subset]["Coefficient Names"])):
                model.coeff_[i] = subresults[best_model_subset][
                    "Estimated Coefficients"
                ][i]
            print(f"Substituting {specification} with results from {best_model_subset}")
        else:
            second_choice_ll = compute_second_choice_likelihood(
                model,
                first_choice_data,
                second_choice_data,
                varnames,
            )
            rmse, predicted_diversion_matrix = compute_second_choice_rmse(
                model,
                first_choice_data,
                empirical_diversion_matrix,
                return_predicted_diversion_matrix=True,
            )
    lr_test = (
        0
        if specification == "plain logit"
        else 2 * (model.loglikelihood - subresults["plain logit"]["First Choice LL"])
    )

    result = {
        "First Choice LL": model.loglikelihood,
        "First Choice AIC": model.aic,
        "Second Choice LL": second_choice_ll,
        "Second Choice RMSE": rmse,
        "Coefficient Names": model.coeff_names,
        "Estimated Coefficients": model.coeff_,
        "Likelihood Ratio Test": stats.chi2.sf(lr_test, 1),
        "Predicted Diversion Matrix": predicted_diversion_matrix,
    }
    print(f"Specification: {specification}. AIC: {model.aic}. RMSE: {rmse}")
    subresults[specification] = result


def main():
    start_time = time.time()

    input_path = "data/experiment/input"

    first_choice_data, second_choice_data, empirical_diversion_matrix, book_titles = (
        load_data_long(input_path)
    )

    product_fixed_effects_varnames = [
        "product_id_3",
        "product_id_4",
        "product_id_8",
        "product_id_9",
        "product_id_21",
        "product_id_23",
        "product_id_29",
        "product_id_45",
        "product_id_46",
        # "product_id_47",
    ]

    varnames = ["price", "position", "year", "pages", "genre_mystery", "genre_scifi"]

    varnames_product_attributes = varnames + product_fixed_effects_varnames

    product_attributes_randvars = {
        "plain logit": {},
        # Single attribute models
        "price": {"price": "n"},
        "pages": {"pages": "n"},
        "year": {"year": "n"},
        "genre": {"genre_mystery": "n", "genre_scifi": "n"},
        # Two attribute models
        "price and pages": {"price": "n", "pages": "n"},
        "price and year": {"price": "n", "year": "n"},
        "price and genre": {"price": "n", "genre_mystery": "n", "genre_scifi": "n"},
        "pages and year": {"pages": "n", "year": "n"},
        "pages and genre": {
            "pages": "n",
            "genre_mystery": "n",
            "genre_scifi": "n",
        },
        "year and genre": {"year": "n", "genre_mystery": "n", "genre_scifi": "n"},
        # Three attribute models
        "price, pages, and year": {"price": "n", "pages": "n", "year": "n"},
        "price, pages, and genre": {
            "price": "n",
            "pages": "n",
            "genre_mystery": "n",
            "genre_scifi": "n",
        },
        "price, year, and genre": {
            "price": "n",
            "year": "n",
            "genre_mystery": "n",
            "genre_scifi": "n",
        },
        "pages, year, and genre": {
            "pages": "n",
            "year": "n",
            "genre_mystery": "n",
            "genre_scifi": "n",
        },
        # Four attribute model
        "price, pages, year, and genre": {
            "price": "n",
            "pages": "n",
            "year": "n",
            "genre_mystery": "n",
            "genre_scifi": "n",
        },
    }

    embedding_models = list(
        set([col.split("_pc")[0] for col in first_choice_data.columns if "_pc" in col])
    )

    results = {model: {} for model in embedding_models}
    results["observables"] = {}

    print("Estimating mixed logit models...")
    for specification, randvars in product_attributes_randvars.items():
        start_time = time.time()

        model = estimate_mixed_logit(
            first_choice_data, varnames_product_attributes, randvars
        )

        save_model_to_results(
            model,
            results["observables"],
            specification,
            first_choice_data,
            second_choice_data,
            varnames_product_attributes,
            empirical_diversion_matrix,
        )

        print(
            f"Time: {(time.time() - start_time) / 60:.2f} minutes for {specification}"
        )

    for i, embedding_model in enumerate(embedding_models):
        print(
            f"Estimating for embedding model: {embedding_model}, {i+1}/{len(embedding_models)}"
        )
        start_time = time.time()
        pc_varnames = (
            product_fixed_effects_varnames
            + ["price", "position"]
            + [f"{embedding_model}_pc{i}" for i in range(1, 4)]
        )
        pc_specifications = {
            "plain logit": {},
            # Single attribute models
            "price": {"price": "n"},
            "PC1": {f"{embedding_model}_pc1": "n"},
            "PC2": {f"{embedding_model}_pc2": "n"},
            "PC3": {f"{embedding_model}_pc3": "n"},
            # Two attribute models
            "price and PC1": {"price": "n", f"{embedding_model}_pc1": "n"},
            "price and PC2": {"price": "n", f"{embedding_model}_pc2": "n"},
            "price and PC3": {"price": "n", f"{embedding_model}_pc3": "n"},
            "PC1 and PC2": {
                f"{embedding_model}_pc1": "n",
                f"{embedding_model}_pc2": "n",
            },
            "PC1 and PC3": {
                f"{embedding_model}_pc1": "n",
                f"{embedding_model}_pc3": "n",
            },
            "PC2 and PC3": {
                f"{embedding_model}_pc2": "n",
                f"{embedding_model}_pc3": "n",
            },
            # Three attribute models
            "price, PC1, and PC2": {
                "price": "n",
                f"{embedding_model}_pc1": "n",
                f"{embedding_model}_pc2": "n",
            },
            "price, PC1, and PC3": {
                "price": "n",
                f"{embedding_model}_pc1": "n",
                f"{embedding_model}_pc3": "n",
            },
            "price, PC2, and PC3": {
                "price": "n",
                f"{embedding_model}_pc2": "n",
                f"{embedding_model}_pc3": "n",
            },
            "PC1, PC2, and PC3": {
                f"{embedding_model}_pc1": "n",
                f"{embedding_model}_pc2": "n",
                f"{embedding_model}_pc3": "n",
            },
            # Four attribute model
            "price, PC1, PC2, and PC3": {
                "price": "n",
                f"{embedding_model}_pc1": "n",
                f"{embedding_model}_pc2": "n",
                f"{embedding_model}_pc3": "n",
            },
        }

        for specification, randvars in pc_specifications.items():
            model = estimate_mixed_logit(first_choice_data, pc_varnames, randvars)

            save_model_to_results(
                model,
                results[embedding_model],
                specification,
                first_choice_data,
                second_choice_data,
                pc_varnames,
                empirical_diversion_matrix,
            )

        print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")

    output_dir = "data/experiment/output/estimation_results"
    todays_date = time.strftime("%Y-%m-%d")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"mixed_logit_results_{todays_date}.xlsx")
    output_path_diversion_matrices = os.path.join(
        output_dir,
        f"mixed_logit_diversion_matrices_{todays_date}.xlsx",
    )

    save_results_to_xlsx(results, output_path)
    save_predicted_diversion_matrices_to_xlsx(
        results, output_path_diversion_matrices, empirical_diversion_matrix, book_titles
    )

    print(f"Results saved to: {output_path}")
    print(f"Diversion matrices saved to: {output_path_diversion_matrices}")

    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
