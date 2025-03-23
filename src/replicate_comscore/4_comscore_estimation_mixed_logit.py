# python -m src.replicate_comscore.4_comscore_estimation_mixed_logit
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# Ignore numpy warnings from xlogit # TODO: fix the warnings?
warnings.filterwarnings("ignore")
from src.helper_functions.estimation.estimate_mixed_logit import estimate_mixed_logit
from src.helper_functions.estimation.load_data import load_data_long_comscore


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
                "Coefficient Names",
                "Estimated Coefficients",
                "Standard Errors",
                "Likelihood Ratio Test",
            ]

            # If the df doesn't have the required columns, skip
            if not set(columns).issubset(df.columns):
                continue

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


def save_model_to_results(
    model,
    subresults,
    specification,
):
    if subresults == {}:
        subresults["plain logit"] = {
            "First Choice LL": model.loglikelihood,
            "First Choice AIC": model.aic,
            "Coefficient Names": model.coeff_names,
            "Estimated Coefficients": model.coeff_,
            "Standard Errors": model.stderr,
            "Likelihood Ratio Test": 0,
        }
        return

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
        model.coeff_ = np.zeros(len(model.coeff_names))
        for i in range(len(subresults[best_model_subset]["Coefficient Names"])):
            model.coeff_[i] = subresults[best_model_subset]["Estimated Coefficients"][i]
    lr_test = (
        0
        if specification == "plain logit"
        else 2 * (model.loglikelihood - subresults["plain logit"]["First Choice LL"])
    )

    result = {
        "First Choice LL": model.loglikelihood,
        "First Choice AIC": model.aic,
        "Coefficient Names": model.coeff_names,
        "Estimated Coefficients": model.coeff_,
        "Standard Errors": model.stderr,
        "Likelihood Ratio Test": stats.chi2.sf(lr_test, 1),
    }
    subresults[specification] = result


def main():

    input_path = "data/comscore/intermediate"

    categories_needed = pd.read_csv(
        "data/comscore/input/comscore_categories.csv", dtype=str
    )["Category_Code"].tolist()
    print(f"Categories needed: {categories_needed}")
    print(f"Number of categories needed: {len(categories_needed)}")

    categories_with_errors = []

    for i, category in enumerate(categories_needed):
        try:
            start_time = time.time()

            print("=" * 40)
            print(
                f"Estimating for category: {category}, {i+1}/{len(categories_needed)}"
            )
            category_path = os.path.join(input_path, category)
            first_choice_data = load_data_long_comscore(category_path)
            product_fixed_effects_varnames_minus_1 = sorted(
                [
                    col
                    for col in first_choice_data.columns
                    if col.startswith("product_id_")
                ]
            )[:-1]
            product_fixed_effects_varnames_minus_4 = sorted(
                [
                    col
                    for col in first_choice_data.columns
                    if col.startswith("product_id_")
                ]
            )[:-4]
            assert len(product_fixed_effects_varnames_minus_4) + 3 == len(
                product_fixed_effects_varnames_minus_1
            )
            embedding_models = sorted(
                list(
                    set(
                        [
                            col.split("_pc")[0]
                            for col in first_choice_data.columns
                            if "_pc" in col
                        ]
                    )
                )
            )

            results = {model: {} for model in embedding_models}

            print("Estimating mixed logit models...")
            for i, embedding_model in enumerate(embedding_models):
                print(
                    f"Estimating for embedding model: {embedding_model}, {i+1}/{len(embedding_models)}"
                )
                pc_varnames = (
                    product_fixed_effects_varnames_minus_4
                    + ["price"]
                    + [f"{embedding_model}_pc{i}" for i in range(1, 4)]
                )
                pc_specifications = {
                    "plain logit with PCs": {},
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

                # NOTE: This model should include the full fixed effects (J-1)
                plain_logit_no_pc_model = estimate_mixed_logit(
                    first_choice_data,
                    product_fixed_effects_varnames_minus_1 + ["price"],
                    {},
                )
                save_model_to_results(
                    plain_logit_no_pc_model,
                    results[embedding_model],
                    "plain logit",
                )

                # NOTE: For the remaining, include J - 4 fixed effects, assert that plain logit with PC means equals full fixed effects model
                for specification, randvars in pc_specifications.items():
                    model = estimate_mixed_logit(
                        first_choice_data, pc_varnames, randvars
                    )
                    if specification == "plain logit with PCs":
                        # assert likelihoods are close to equal
                        if (
                            model.loglikelihood
                            and plain_logit_no_pc_model.loglikelihood
                        ):
                            if (
                                abs(
                                    model.loglikelihood
                                    - plain_logit_no_pc_model.loglikelihood
                                )
                                > 1e-4
                            ):
                                print(
                                    f"Likelihoods are not close for plain logit with PCs and plain logit: {model.loglikelihood} vs {plain_logit_no_pc_model.loglikelihood}"
                                )

                    save_model_to_results(
                        model,
                        results[embedding_model],
                        specification,
                    )

            output_dir = "data/comscore/output/estimation_results"
            todays_date = time.strftime("%Y-%m-%d")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"mixed_logit_results_{category}_{todays_date}.xlsx"
            )

            save_results_to_xlsx(results, output_path)

            print(f"Results saved to: {output_path}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time elapsed: {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"Error for category: {category}")
            print(e)
            categories_with_errors.append(category)

    print("=" * 40)
    print("Categories with errors:")
    print(categories_with_errors)


if __name__ == "__main__":
    main()
