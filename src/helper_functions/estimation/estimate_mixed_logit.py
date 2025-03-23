import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from xlogit import MixedLogit


def standardize(series):
    return (series - series.mean()) / series.std()


def load_data_long(input_path, save_long_data=False):
    # Paths to the relevant data
    choice_data_path = (
        "data/experiment/intermediate/survey_responses/ebook_mult_logit_data.csv"
    )
    book_data_path = os.path.join(input_path, "books/books.csv")
    book_principal_components_path = "data/experiment/intermediate/principal_components"

    # Load book data
    book_data = pd.read_csv(book_data_path)
    product_id_to_asin = dict(zip(book_data["id"], book_data["asin"]))
    sorted_asins = sorted(book_data["asin"])
    book_titles = book_data["title"].values

    # Step 2: Load principal components into a dictionary
    principal_components = {}

    for file in os.listdir(book_principal_components_path):
        if file.endswith("_principal_components.csv"):
            # if file.endswith("_3PCs.csv"):
            model = file.split("_principal_components")[0]
            # model = file.split("_3PCs")[0]
            filepath = os.path.join(book_principal_components_path, file)

            # Load the CSV and assign the sorted asin as index
            pc_data = pd.read_csv(filepath)
            pc_data.index = sorted_asins  # Use sorted asin order

            # Add each principal component column to the dictionary
            for i, col in enumerate(pc_data.columns):
                if col == "asin":
                    continue
                # NOTE: Normalize each principal component
                pc_normalized = standardize(pc_data[col])
                principal_components[f"{model}_pc{i}"] = dict(
                    zip(pc_data.index, pc_normalized)
                )

    # Step 3: Load choice data
    choice_data = pd.read_csv(choice_data_path)

    # Step 4: Merge principal components into choice_data
    # Map product_id to asin for choice_data
    choice_data["asin"] = choice_data["product_id"].map(
        lambda x: product_id_to_asin.get(x)
    )

    # Add columns for each principal component
    for key, pc_dict in principal_components.items():
        choice_data[key] = choice_data["asin"].map(pc_dict)

    # Step 5: Map additional book attributes into choice_data
    book_data["publication_year_normalized"] = standardize(
        book_data["publication_year"]
    )
    book_data["number_of_pages_normalized"] = standardize(book_data["number_of_pages"])
    # book_data["publication_year_normalized"] = book_data["publication_year"]
    # book_data["number_of_pages_normalized"] = book_data["number_of_pages"]

    year_map_normalized = dict(
        zip(book_data["asin"], book_data["publication_year_normalized"])
    )
    pages_map_normalized = dict(
        zip(book_data["asin"], book_data["number_of_pages_normalized"])
    )
    genre_mystery_map = dict(
        zip(
            book_data["asin"],
            (book_data["genre"] == "Mystery, Thriller & Suspense").astype(int),
        )
    )
    genre_scifi_map = dict(
        zip(
            book_data["asin"],
            (book_data["genre"] == "Science Fiction & Fantasy").astype(int),
        )
    )

    choice_data["year"] = choice_data["asin"].map(
        lambda x: year_map_normalized.get(x, None)
    )
    choice_data["pages"] = choice_data["asin"].map(
        lambda x: pages_map_normalized.get(x, None)
    )
    choice_data["genre_mystery"] = choice_data["asin"].map(
        lambda x: genre_mystery_map.get(x, 0)
    )
    choice_data["genre_scifi"] = choice_data["asin"].map(
        lambda x: genre_scifi_map.get(x, 0)
    )

    choice_data["product_id_4"] = choice_data["product_id"].map(
        lambda x: 1 if x == 4 else 0
    )

    # Save the long data to a CSV
    if save_long_data:
        output_path = os.path.join(input_path, "survey_responses/long_data.csv")
        choice_data.to_csv(output_path, index=False)
        print(f"Long data saved to: {output_path}")

    # Split into first and second choice data
    first_choice_data = choice_data[choice_data["choice_number"] == 1]
    second_choice_data = choice_data[choice_data["choice_number"] == 2]

    # Compute empirical diversion matrix
    J = len(choice_data["product_id"].unique())
    count_matrix = np.zeros((J, J), dtype=float)

    # entry (i, j) is the number of times product i was second choice when product j was first choice
    # count_matrix[second_choice_idx, first_choice_idx] += 1
    product_ids = sorted(choice_data["product_id"].unique())

    for consumer_id in choice_data["respondent_id"].unique():  # 9265 consumers
        first_choice = first_choice_data[
            (first_choice_data["respondent_id"] == consumer_id)
            & (first_choice_data["choice"] == 1)
        ]["product_id"].values[0]
        second_choice = second_choice_data[
            (second_choice_data["respondent_id"] == consumer_id)
            & (second_choice_data["choice"] == 1)
        ]["product_id"].values[0]

        first_choice_idx = product_ids.index(first_choice)
        second_choice_idx = product_ids.index(second_choice)

        count_matrix[first_choice_idx, second_choice_idx] += 1

    # normalize the count_matrix
    empirical_diversion_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)

    return (
        first_choice_data,
        second_choice_data,
        empirical_diversion_matrix,
        book_titles,
    )


def _fit_single_model(random_state, data, varnames, randvars, n_draws, halton=False):
    """Helper function to fit a single model with given random state"""
    try:
        model = MixedLogit()
        model.fit(
            X=data[varnames],
            y=data["choice"],
            varnames=varnames,
            ids=data["choice_id"],
            alts=data["product_id"],
            n_draws=n_draws,
            random_state=random_state,
            randvars=randvars,
            halton=halton,
            verbose=0,
        )
        return model, model.loglikelihood
    except Exception as e:
        print(f"Warning: Fitting failed for random_state {random_state}: {str(e)}")
        return None, -np.inf


def estimate_mixed_logit(
    data,
    varnames,
    randvars,
    n_draws=100,
    num_starting_points=100,
    seed=1,
    return_empty=False,
    halton=False,
):
    """
    Estimate mixed logit model with multiple random starting points in parallel.
    Returns the model with the highest log-likelihood.

    Args:
        data (pd.DataFrame): Input data
        varnames (list): List of variable names
        randvars (dict): Random variables specification
        n_draws (int): Number of draws for simulation
        num_starting_points (int): Number of different random starting points to try

    Returns:
        MixedLogit: Best fitted model
    """

    if return_empty:
        empty_model = MixedLogit()
        empty_model.loglikelihood = -np.inf
        empty_model.aic = np.inf
        empty_model.coeff_ = np.zeros(len(varnames) + len(randvars))
        sd_coeff_names = [f"sd.{var}" for var in randvars.keys()]
        empty_model.coeff_names = varnames + sd_coeff_names
        empty_model.n_draws = n_draws
        return empty_model

    # Create partial function with fixed arguments
    fit_func = partial(
        _fit_single_model,
        data=data,
        varnames=varnames,
        randvars=randvars,
        n_draws=n_draws,
        halton=halton,
    )

    n_cores = max(1, cpu_count() - 1)

    # Create pool of workers
    with Pool(n_cores) as pool:
        # Run fits in parallel with different random states
        results = pool.map(
            fit_func,
            range(
                1 + num_starting_points * seed,
                num_starting_points + 1 + num_starting_points * seed,
            ),
        )

    # Filter out failed fits and find best model
    valid_results = [(model, ll) for model, ll in results if model is not None]

    if not valid_results:
        raise RuntimeError(
            "All model fits failed. Please check your data and parameters."
        )

    # Get model with highest log-likelihood
    best_model, _ = max(valid_results, key=lambda x: x[1])

    return best_model


def predict_mixed_logit(model, data, varnames, avail=None):
    _, predicted_probs = model.predict(
        X=data[varnames],
        varnames=varnames,
        ids=data["choice_id"],
        alts=data["product_id"],
        avail=avail,
        return_proba=True,
        halton=False,
        random_state=1,  # Fix random_state
    )

    return predicted_probs


def compute_second_choice_likelihood(
    model,
    first_choice_data,
    second_choice_data,
    varnames,
    return_0=False,
):
    """
    Computes the Second Choice Log Likelihood based on the model's predictions.

    Parameters:
    - model: Trained MixedLogit model with a predict method.
    - first_choice_data (pd.DataFrame): DataFrame containing first choice information.
    - second_choice_data (pd.DataFrame): DataFrame containing second choice information.
    - varnames (list): List of variable names used in the model.

    Returns:
    - second_choice_ll (float): Computed log-likelihood for second choices.
    """
    if return_0:
        return 0
    # Predict original market shares (probabilities) for the first choice
    mkt_shares = predict_mixed_logit(model, first_choice_data, varnames)

    # Predict market shares without the first choice
    first_choice_removed = first_choice_data.apply(
        lambda row: 0 if row["choice"] == 1 else 1,
        axis=1,
    )

    new_mkt_shares = predict_mixed_logit(
        model, first_choice_data, varnames, avail=first_choice_removed
    )

    # Extract first and second choice indices
    unique_products = first_choice_data["product_id"].unique()
    product_to_index = {product: idx for idx, product in enumerate(unique_products)}

    # Map `product_id` to indices for first and second choices
    first_choices = (
        first_choice_data[first_choice_data["choice"] == 1]
        .loc[:, "product_id"]
        .map(product_to_index)
        .to_numpy()
    )

    second_choices = (
        second_choice_data[second_choice_data["choice"] == 1]
        .loc[:, "product_id"]
        .map(product_to_index)
        .to_numpy()
    )

    # Extract the predicted probabilities
    s_j = mkt_shares[np.arange(len(mkt_shares)), first_choices]
    s_k = mkt_shares[np.arange(len(mkt_shares)), second_choices]
    s_prime_k = new_mkt_shares[np.arange(len(new_mkt_shares)), second_choices]

    # Compute metric
    # To avoid division by zero, add a small epsilon
    epsilon = 1e-10
    metric = (s_prime_k - s_k) / (s_j + epsilon)

    # Compute log metric, handle non-positive metrics
    # Replace non-positive metrics with a small positive number before taking log
    metric = np.where(metric > 0, metric, epsilon)
    log_metric = np.log(metric)

    # Sum the log metrics
    second_choice_ll = np.sum(log_metric)

    return second_choice_ll


def simulate_individual(
    first_choice_data,
    coeff_dict,
    i,
    seed=1,
    include_gumbels=True,
):
    unique_products = first_choice_data["product_id"].unique()
    J = len(unique_products)

    np.random.seed(seed)
    # Extract the subset of rows corresponding to this individual's choice situation
    df_i = first_choice_data.loc[first_choice_data["choice_id"] == i]

    # Draw individual-specific random coefficients (if any)
    individual_coeffs = {}
    for param in coeff_dict.keys():
        # Skip the 'sd_' parameters themselves; these are used for the standard deviation
        # of the random parameters
        if param.startswith("sd."):
            param_mean = param[3:]
            mean_ = coeff_dict[param_mean]
            sd_ = np.abs(coeff_dict[param])
            individual_coeffs[param_mean] = np.random.normal(mean_, sd_)
        else:
            # Otherwise, it's a fixed parameter
            individual_coeffs[param] = coeff_dict[param]

    # Compute utilities for each product
    # Add Gumbel random error for each product as well
    utilities = []
    if include_gumbels:
        epsilons = np.random.gumbel(loc=0, scale=1, size=J)
    else:
        epsilons = np.zeros(J)

    # Sort df_i by product_id to ensure consistent ordering
    df_i = df_i.sort_values(by="product_id")
    df_i_indexed = df_i.set_index("product_id")

    # For each product in unique_products (which are sorted), compute the utility
    for j, j_id in enumerate(unique_products):
        row_ij = df_i_indexed.loc[j_id]

        util_ij = 0.0
        # Sum up the utility from each variable in varnames
        for v in individual_coeffs.keys():
            if v.startswith("sd."):
                continue
            if v in df_i.columns:
                util_ij += individual_coeffs[v] * row_ij[v]
            else:
                raise ValueError(f"Variable {v} not found in df_i.columns")

        # Add the random Gumbel error
        util_ij += epsilons[j]

        utilities.append(util_ij)

    utilities = np.array(utilities)

    return utilities


def generate_predicated_diversion_matrix(coeff_dict, first_choice_data, seed=1):
    np.random.seed(seed)
    choice_ids = first_choice_data["choice_id"].unique()
    unique_products = first_choice_data["product_id"].unique()
    J = len(unique_products)

    # Prepare an empty count matrix: (J x J).
    # predicted_count_matrix[j1, j2] = number of times product j2 is chosen second
    # given that product j1 was chosen first.
    predicted_count_matrix = np.zeros((J, J), dtype=float)

    # Loop through each choice situation
    for i in choice_ids:
        utilities = simulate_individual(first_choice_data, coeff_dict, i, seed=seed + i)
        sorted_idx = np.argsort(utilities)[::-1]
        first_choice_idx = sorted_idx[0]
        second_choice_idx = sorted_idx[1]

        # Map these back to actual product IDs
        first_choice_product_id = unique_products[first_choice_idx]
        second_choice_product_id = unique_products[second_choice_idx]

        # Update predicted_count_matrix
        row_index_for_first = np.where(unique_products == first_choice_product_id)[0][0]
        col_index_for_second = np.where(unique_products == second_choice_product_id)[0][
            0
        ]
        predicted_count_matrix[row_index_for_first, col_index_for_second] += 1

    # Compute predicted diversion matrix
    col_sums = predicted_count_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_diversion_matrix = np.where(
            col_sums != 0, predicted_count_matrix / col_sums, 0
        )

    assert np.allclose(predicted_diversion_matrix.sum(axis=1), 1)

    return predicted_diversion_matrix


def compute_second_choice_rmse(
    model,
    first_choice_data,
    empirical_diversion_matrix,
    seed=123,
    return_predicted_diversion_matrix=False,
):
    """
    Computes the predicted diversion matrix by:
      1) Using the model's estimated parameters (both fixed and random) to compute
         predicted utilities for each individual-product pair.
      2) Determining each individual's first and second choice.
      3) Constructing the predicted diversion matrix from these second choices.
      4) Computing the RMSE of the predicted diversion matrix vs. the empirical one.

    Parameters
    ----------
    model : an object containing, at least:
    first_choice_data : pd.DataFrame
    varnames : list of str
    randvars : dict
    empirical_diversion_matrix : np.ndarray

    Returns
    -------
    rmse : float
    """
    # Create a dictionary of parameter names -> parameter values
    coeff_dict = dict(zip(model.coeff_names, model.coeff_))

    predicted_diversion_matrix = generate_predicated_diversion_matrix(
        coeff_dict, first_choice_data, seed=seed
    )

    # assert that all the rows sum to 1
    assert np.allclose(predicted_diversion_matrix.sum(axis=1), 1)
    assert np.allclose(empirical_diversion_matrix.sum(axis=1), 1)

    # Compute RMSE
    rmse = np.sqrt(
        np.mean((empirical_diversion_matrix - predicted_diversion_matrix) ** 2)
    )

    if return_predicted_diversion_matrix:
        return rmse, predicted_diversion_matrix

    return rmse
