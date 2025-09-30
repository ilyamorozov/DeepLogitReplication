# python -m src.replicate_experiment.3_experiment_estimation_mixed_logit

import os
import time

import numpy as np
from scipy import stats

from src.helper_functions.estimation.estimate_mixed_logit import (
    compute_second_choice_likelihood,
    compute_second_choice_rmse_mae,
    estimate_mixed_logit,
    load_data_long,
)

from src.helper_functions.estimation.gen_specs import gen_specs

from src.helper_functions.visualization.save_results import (
    save_results_to_xlsx,
    save_predicted_diversion_matrices_to_xlsx
)

from src.helper_functions.file_structure.get_file_path_from_config import get_file_path_from_config

def save_model_to_results(
    model,
    subresults,
    specification,
    first_choice_data,
    second_choice_data,
    varnames,
    empirical_diversion_matrix,
    step = None,
):
    """
    Save model results to the subresults dictionary.
    Arguments:
        model: The estimated mixed logit model.
        subresults: Dictionary to store results of different model specifications.
        specification: Name of the current model specification.
        first_choice_data: DataFrame containing first choice data.
        second_choice_data: DataFrame containing second choice data.
        varnames: List of variable names used in the model.
        empirical_diversion_matrix: Empirical diversion matrix for second choice evaluation.
        step: Current step in the augmented set algorithm. (default is None, shouldn't be used for saving observables)
    Returns:
        None. Updates subresults in place.
    """

    if subresults == {}:
        second_choice_ll, s_unconditional = compute_second_choice_likelihood(
            model,
            first_choice_data,
            second_choice_data,
            varnames,
        )
        rmse, mae, predicted_diversion_matrix = compute_second_choice_rmse_mae(
            first_choice_data,
            s_unconditional,
            model,
            varnames,
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
            model.aic = 2 * (
                len(model.coeff_names) - model.loglikelihood
            )  # Recompute AIC using number of parameters of richer model
            second_choice_ll = subresults[best_model_subset]["Second Choice LL"]
            rmse = subresults[best_model_subset]["Second Choice RMSE"]
            mae = subresults[best_model_subset]["Second Choice MAE"]
            predicted_diversion_matrix = subresults[best_model_subset][
                "Predicted Diversion Matrix"
            ]
            model.coeff_ = np.zeros(len(model.coeff_names))
            for i, name in enumerate(model.coeff_names):
                if name in subresults[best_model_subset]["Coefficient Names"]:
                    model.coeff_[i] = subresults[best_model_subset][
                        "Estimated Coefficients"
                    ][
                        np.where(
                            subresults[best_model_subset]["Coefficient Names"] == name
                        )[0][0]
                    ]
            print(f"Substituting {specification} with results from {best_model_subset}")
        else:
            second_choice_ll, s_unconditional = compute_second_choice_likelihood(
                model,
                first_choice_data,
                second_choice_data,
                varnames,
            )
            rmse, mae, predicted_diversion_matrix = compute_second_choice_rmse_mae(
                first_choice_data,
                s_unconditional,
                model,
                varnames,
                empirical_diversion_matrix,
                return_predicted_diversion_matrix=True,
            )
    lr_test = (
        0
        if specification == "plain logit"
        else 2 * (model.loglikelihood - subresults["plain logit"]["First Choice LL"])
    )

    # Number of random SD parameters (additional vs. baseline plain logit
    df = sum("sd" in name for name in model.coeff_names)

    result = {
        "First Choice LL": model.loglikelihood,
        "First Choice AIC": model.aic,
        "Second Choice LL": second_choice_ll,
        "Second Choice RMSE": rmse,
        "Second Choice MAE": mae,
        "Coefficient Names": model.coeff_names,
        "Estimated Coefficients": model.coeff_,
        "Likelihood Ratio Test": stats.chi2.sf(lr_test, df) if df > 0 else 1.0,
        "Predicted Diversion Matrix": predicted_diversion_matrix,
    }
    if step is not None:
        result["Step"] = step
        print(f"Specification: {specification}. AIC: {model.aic}. RMSE: {rmse}. MAE: {mae}. Step: {step}")
    else:
        print(f"Specification: {specification}. AIC: {model.aic}. RMSE: {rmse}. MAE: {mae}")

    subresults[specification] = result


def main():
    # Estimate over K principal components
    k = 6

    start_time = time.time()

    # input_path = "data/experiment/input"
    input_path = get_file_path_from_config(path_type="EXPERIMENT_3", path="INPUT_PATH")

    #book_principal_components_path = "data/experiment/intermediate/principal_components/"
    book_principal_components_path = get_file_path_from_config(path_type="EXPERIMENT_3", path="BOOK_PRINCIPAL_COMPONENTS_PATH")

    first_choice_data, second_choice_data, empirical_diversion_matrix, book_titles = (
        load_data_long(input_path, book_principal_components_path)
    )

    output_dir = get_file_path_from_config(path_type="EXPERIMENT_3", path="OUTPUT_DIR")
    diversion_matrices_dir = get_file_path_from_config(path_type="EXPERIMENT_3", path="OUTPUT_DIR_DIVERSION_MATRICES")
    todays_date = time.strftime("%Y-%m-%d")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(diversion_matrices_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"mixed_logit_results_{todays_date}.xlsx")
    output_path_diversion_matrices = os.path.join(
        diversion_matrices_dir,
        f"mixed_logit_diversion_matrices_{todays_date}.xlsx",
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

    product_attributes_randvars = gen_specs(observables=True)

    embedding_models = list(
        set([col.split("_pc")[0] for col in first_choice_data.columns if "_pc" in col])
    )

    results = {model: {} for model in embedding_models}
    results["observables"] = {}

    best_spec_dict = {model: "" for model in embedding_models}
    best_spec_dict["observables"] = ""

    best_aic_obs = float("inf")
    print("Estimating mixed logit models...")
    for specification, randvars in product_attributes_randvars.items():
        start_time = time.time()

        model = estimate_mixed_logit(
            first_choice_data, varnames_product_attributes, randvars,
            limit_cores=True
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

        if results["observables"][specification]["First Choice AIC"] < best_aic_obs:
            best_aic_obs = results["observables"][specification]["First Choice AIC"]
            best_spec_dict["observables"] = specification

        print(
            f"Time: {(time.time() - start_time) / 60:.2f} minutes for {specification}"
        )

    save_results_to_xlsx(results, output_path, best_spec_dict)
    save_predicted_diversion_matrices_to_xlsx(
        results, output_path_diversion_matrices, empirical_diversion_matrix, book_titles, best_spec_dict  
    )

    print(f"Results for Observables saved to: {output_path}")
    print(f"Diversion matrix for Observables saved to: {output_path_diversion_matrices}")

    for i, embedding_model in enumerate(embedding_models):
        # Generate all specifications of length up to k
        pc_specifications = gen_specs(k, embedding_model)

        print(
            f"Estimating for embedding model: {embedding_model}, {i+1}/{len(embedding_models)}"
        )
        start_time = time.time()
        pc_varnames = (
            product_fixed_effects_varnames
            + ["price", "position"]
            + [f"{embedding_model}_pc{i}" for i in range(1, k+1)]
        )

        # Keep track of best AIC, if best AIC for step j is >= best AIC for step j-1,
        # continue estimating but do not update best specification
        best_aic = float("inf")
        aic_improved = True
        aic_update_allowed = True

        # This loop represent the steps in the augmented set algorithm
        # For each step, we first check if the model has already been estimated in a previous step
        # If it has, we copy the results over to the current step
        # If not, we estimate the model and save the results
        # We are estimating from step 0 (just plain logit) to step k+1 (all k PCs and price)
        for j in range(0, k+2):
            print(f"Step {j} for embedding model {embedding_model}")

            # Get all specifications with j random variables
            pc_specifications_step_j = {
                spec: randvars
                for spec, randvars in pc_specifications.items()
                if spec.count("PC") + spec.count("price") == j
            }

            # Set aic_improved to False at the start of each step of the algorithm
            # We only move to next iteration if we set it to True in the below loop
            aic_improved = False 

            for specification, randvars in pc_specifications_step_j.items():
                model = estimate_mixed_logit(first_choice_data, pc_varnames, randvars, limit_cores=True)

                save_model_to_results(
                    model=model,
                    subresults=results[embedding_model],
                    specification=specification,
                    first_choice_data=first_choice_data,
                    second_choice_data=second_choice_data,
                    varnames=pc_varnames,
                    empirical_diversion_matrix=empirical_diversion_matrix,
                    step=j,
                )
                if results[embedding_model][specification]["First Choice AIC"] < best_aic and aic_update_allowed:
                    best_aic = results[embedding_model][specification]["First Choice AIC"]
                    best_spec_dict[embedding_model] = specification
                    aic_improved = True
            # When below condition is met, we will be at the step following the one from which we take our selected model
            # if we don't achieve improvement in AIC in this step, but we are still allowed to update aic
            # we should set this to aic_update_allowed to False and display the below message and then continue with estimation without updates to selected model 
            if not aic_improved and aic_update_allowed:
                print(f"No AIC improvement in step {j}, estimation continuing without updates to best specification")
                aic_update_allowed = False

        save_results_to_xlsx(results, output_path, best_spec_dict)
        save_predicted_diversion_matrices_to_xlsx(
            results, output_path_diversion_matrices, empirical_diversion_matrix, book_titles, best_spec_dict  
        )
        #free up memory in results
        results[embedding_model] = {}

        print(f"Results for embedding model {embedding_model} saved to: {output_path}")
        print(f"Diversion matrix for embedding model {embedding_model} saved to: {output_path_diversion_matrices}")

        print(f"Time: {(time.time() - start_time) / 60:.2f} minutes")

    print(f"Results saved to: {output_path}")
    print(f"Diversion matrices saved to: {output_path_diversion_matrices}")

    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
