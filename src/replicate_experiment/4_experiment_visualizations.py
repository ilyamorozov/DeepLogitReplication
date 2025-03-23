# python -m src.replicate_experiment.4_experiment_visualizations

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

###############################################################################
# 1) Define a user-friendly mapping for model names (from tab names, specs, etc.)
###############################################################################
formatted_submodel_map = {
    # Benchmarks:
    "plain_logit": "Plain Logit",
    "observables": "Observables",
    # Images:
    "xception_image": "Xception",
    "inceptionv3_image": "InceptionV3",
    "vgg19_image": "VGG19",
    "resnet50_image": "ResNet50",
    "vgg16_image": "VGG16",
    # Titles:
    "title_TFIDF_text": "Bag-of-Words TF-IDF",
    "title_ST_text": "Sentence Transformer (ST)",
    "title_USE_text": "Sentence Encoder (USE)",
    "title_COUNT_text": "Bag-of-Words Count",
    # Descriptions:
    "description_TFIDF_text": "Bag-of-Words TF-IDF",
    "description_ST_text": "Sentence Transformer (ST)",
    "description_USE_text": "Sentence Encoder (USE)",
    "description_COUNT_text": "Bag-of-Words Count",
    # User Reviews:
    "user_reviews_COUNT_text": "Bag-of-Words Count",
    "user_reviews_USE_text": "Sentence Encoder (USE)",
    "user_reviews_TFIDF_text": "Bag-of-Words TF-IDF",
    "user_reviews_ST_text": "Sentence Transformer (ST)",
    # Combined:
    "combined": "Combined",
    # Best PCA:
    "best_pca": "Best Mixed Logit (Texts)",
    "price": "Price",
    "pages": "Pages",
    "year": "Year",
    "genre": "Genre",
    "price and pages": "Price & Pages",
    "price and year": "Price & Year",
    "pages and year": "Pages & Year",
    "price and genre": "Price & Genre",
    "pages and genre": "Pages & Genre",
    "year and genre": "Year & Genre",
    "price, pages, and year": "Price, Pages, & Year",
    "price, pages, and genre": "Price, Pages, & Genre",
    "price, year, and genre": "Price, Year, & Genre",
    "pages, year, and genre": "Pages, Year, & Genre",
    "price, pages, year, and genre": "All Attributes",
}


###############################################################################
# 2) Helper function: classify a tab name into subheading
###############################################################################
def classify_subheading(tab_name: str) -> str:
    """
    Classifies the Excel tab name into a subheading:
      - "observables" sheet => "Benchmarks" (if it has the plain logit or other)
      - if 'title' in tab name => "Product Titles"
      - if 'image' in tab name => "Product Images"
      - if 'description' in tab name => "Product Descriptions"
      - if 'user_reviews' in tab name => "User Reviews"
      Otherwise => you can add more logic or default to "Other"
    """
    tab_lower = tab_name.lower()

    if "observables" in tab_lower:
        return "Benchmarks"
    elif "title" in tab_lower:
        return "Product Titles"
    elif "image" in tab_lower:
        return "Product Images"
    elif "description" in tab_lower:
        return "Product Descriptions"
    elif "user_reviews" in tab_lower:
        return "User Reviews"
    else:
        return "Other"


###############################################################################
# 3) Read Excel, identify best models, build dictionaries
###############################################################################
def read_excel_and_select_best_models(
    excel_path: str,
):
    """
    Reads the Excel file with multiple tabs. For each tab:
      - Load the data
      - Find the row with best AIC
      - Find the row with best RMSE
      - Insert into the best_aic_dict and best_rmse_dict

    Each dictionary is of the form:
        {
            "Benchmarks": {
                "model_name": {
                    "first_choice_ll": float,
                    "first_choice_aic": float,
                    "second_choice_ll": float,
                    "second_choice_rmse": float,
                    "delta_rmse": float,
                    "percent_delta_rmse": float,
                },
                ...
            },
            "Product Images": { ... },
            "Product Titles": { ... },
            ...
        }
    """
    # Read all sheets at once
    all_sheets = pd.read_excel(excel_path, sheet_name=None)

    # Prepare dictionaries
    best_aic_dict = {}
    best_rmse_dict = {}

    # Initialize all subheadings with empty dicts
    all_possible_subheadings = [
        "Benchmarks",
        "Product Images",
        "Product Titles",
        "Product Descriptions",
        "User Reviews",
        "Other",
    ]
    for subh in all_possible_subheadings:
        best_aic_dict[subh] = {}
        best_rmse_dict[subh] = {}

    # Before iterating all_sheet, sort the keys to ensure a consistent order
    # use the formatted_submodel_map to sort the keys (e.g. the sorted order should be based on the formatted names (the values in the formatted_submodel_map))
    all_sheets = dict(
        sorted(all_sheets.items(), key=lambda x: formatted_submodel_map.get(x[0], x[0]))
    )

    # First get the plain logit model RMSE from the "observables" tab
    # We will use this as a benchmark for all other models
    if "observables" in all_sheets:
        obs_df = all_sheets["observables"]
        row_plain_logit = obs_df[obs_df["Specification"] == "plain logit"]
        if not row_plain_logit.empty:
            row_plain_logit = row_plain_logit.iloc[0]
            plain_logit_rmse = row_plain_logit["Second Choice RMSE"]
        else:
            raise ValueError("No 'plain logit' found in the 'observables' tab")
    else:
        raise ValueError("No 'observables' tab found in the Excel file")

    # For each sheet (tab), pick best row by AIC and best by RMSE
    for tab_name, df in all_sheets.items():
        if "Specification" not in df.columns or tab_name == "observables":
            continue  # skip sheets without the required columns
        if tab_name == "vgg16_image":
            continue  # skip VGG16

        # Ensure required columns exist
        required_cols = [
            "Specification",
            "First Choice LL",
            "First Choice AIC",
            "Second Choice LL",
            "Second Choice RMSE",
        ]
        for col in required_cols:
            if col not in df.columns:
                continue  # or raise an error

        # Identify subheading
        subheading = classify_subheading(tab_name)

        # best AIC row
        aic_min_idx = df["First Choice AIC"].idxmin()
        best_aic_row = df.loc[aic_min_idx]

        # best RMSE row
        rmse_min_idx = df["Second Choice RMSE"].idxmin()
        best_rmse_row = df.loc[rmse_min_idx]

        best_aic_model_key = str(tab_name)
        best_rmse_model_key = str(tab_name)

        best_aic_dict[subheading][best_aic_model_key] = {
            "specification": best_rmse_row["Specification"],
            "first_choice_ll": float(best_aic_row["First Choice LL"]),
            "first_choice_aic": float(best_aic_row["First Choice AIC"]),
            "second_choice_ll": float(best_aic_row["Second Choice LL"]),
            "second_choice_rmse": float(best_aic_row["Second Choice RMSE"]),
            "delta_rmse": float(best_aic_row["Second Choice RMSE"]) - plain_logit_rmse,
            "percent_delta_rmse": (
                float(best_aic_row["Second Choice RMSE"]) - plain_logit_rmse
            )
            / plain_logit_rmse,
        }

        best_rmse_dict[subheading][best_rmse_model_key] = {
            "specification": best_rmse_row["Specification"],
            "first_choice_ll": float(best_rmse_row["First Choice LL"]),
            "first_choice_aic": float(best_rmse_row["First Choice AIC"]),
            "second_choice_ll": float(best_rmse_row["Second Choice LL"]),
            "second_choice_rmse": float(best_rmse_row["Second Choice RMSE"]),
            "delta_rmse": float(best_rmse_row["Second Choice RMSE"]) - plain_logit_rmse,
            "percent_delta_rmse": (
                float(best_rmse_row["Second Choice RMSE"]) - plain_logit_rmse
            )
            / plain_logit_rmse,
        }
    attribute_based_mixed_dict = {}
    attribute_based_mixed_dict["Benchmarks"] = {}
    attribute_based_mixed_dict["Attribute-Based Mixed Logit"] = {}

    # Now, handle the plain logit from "observables" tab for Benchmarks
    # We look for the row in the "observables" tab with "Specification" == "plain logit"
    if "observables" in all_sheets:
        obs_df = all_sheets["observables"]
        # find plain logit
        row_plain_logit = obs_df[obs_df["Specification"] == "plain logit"]
        if not row_plain_logit.empty:
            row_plain_logit = row_plain_logit.iloc[0]
            # Insert into best_aic_dict and best_rmse_dict under Benchmarks
            best_aic_dict["Benchmarks"]["plain_logit"] = {
                "specification": "plain logit",
                "first_choice_ll": float(row_plain_logit["First Choice LL"]),
                "first_choice_aic": float(row_plain_logit["First Choice AIC"]),
                "second_choice_ll": float(row_plain_logit["Second Choice LL"]),
                "second_choice_rmse": float(row_plain_logit["Second Choice RMSE"]),
                "delta_rmse": float(row_plain_logit["Second Choice RMSE"])
                - plain_logit_rmse,
                "percent_delta_rmse": (
                    float(row_plain_logit["Second Choice RMSE"]) - plain_logit_rmse
                )
                / plain_logit_rmse,
            }
            best_rmse_dict["Benchmarks"]["plain_logit"] = {
                "specification": "plain logit",
                "first_choice_ll": float(row_plain_logit["First Choice LL"]),
                "first_choice_aic": float(row_plain_logit["First Choice AIC"]),
                "second_choice_ll": float(row_plain_logit["Second Choice LL"]),
                "second_choice_rmse": float(row_plain_logit["Second Choice RMSE"]),
                "delta_rmse": float(row_plain_logit["Second Choice RMSE"])
                - plain_logit_rmse,
                "percent_delta_rmse": (
                    float(row_plain_logit["Second Choice RMSE"]) - plain_logit_rmse
                )
                / plain_logit_rmse,
            }

        for _, row in obs_df.iterrows():
            if row["Specification"] == "plain logit":
                attribute_based_mixed_dict["Benchmarks"]["plain_logit"] = {
                    "specification": "plain logit",
                    "first_choice_ll": float(row["First Choice LL"]),
                    "first_choice_aic": float(row["First Choice AIC"]),
                    "second_choice_ll": float(row["Second Choice LL"]),
                    "second_choice_rmse": float(row["Second Choice RMSE"]),
                    "delta_rmse": float(row["Second Choice RMSE"]) - plain_logit_rmse,
                    "percent_delta_rmse": (
                        float(row["Second Choice RMSE"]) - plain_logit_rmse
                    )
                    / plain_logit_rmse,
                }
            else:
                attribute_based_mixed_dict["Attribute-Based Mixed Logit"][
                    row["Specification"]
                ] = {
                    "specification": row["Specification"],
                    "first_choice_ll": float(row["First Choice LL"]),
                    "first_choice_aic": float(row["First Choice AIC"]),
                    "second_choice_ll": float(row["Second Choice LL"]),
                    "second_choice_rmse": float(row["Second Choice RMSE"]),
                    "delta_rmse": float(row["Second Choice RMSE"]) - plain_logit_rmse,
                    "percent_delta_rmse": (
                        float(row["Second Choice RMSE"]) - plain_logit_rmse
                    )
                    / plain_logit_rmse,
                }

    # add the best mixed logit model to the attribute_based_mixed_dict["Benchmarks"]
    # we need to find the best mixed logit model from best_rmse_dict across all subheadings
    best_mixed_logit_model = None
    best_subheading = None
    best_mixed_logit_rmse = float("inf")
    best_specification = None
    best_metric = "first_choice_aic"
    for subheading, model_dict in best_rmse_dict.items():
        for model_name, metrics in model_dict.items():
            if metrics[best_metric] < best_mixed_logit_rmse:
                best_mixed_logit_rmse = metrics[best_metric]
                best_mixed_logit_model = model_name
                best_subheading = subheading
                best_specification = metrics["specification"]

    print(f"Best mixed logit model: {best_mixed_logit_model}")
    print(f"Best subheading: {best_subheading}")
    print(f"Best specification: {best_specification}")
    best_mixed_logit_model = None
    best_subheading = None
    best_mixed_logit_rmse = float("inf")
    best_specification = None
    best_metric = "second_choice_rmse"
    for subheading, model_dict in best_rmse_dict.items():
        for model_name, metrics in model_dict.items():
            if metrics[best_metric] < best_mixed_logit_rmse:
                best_mixed_logit_rmse = metrics[best_metric]
                best_mixed_logit_model = model_name
                best_subheading = subheading
                best_specification = metrics["specification"]

    print(f"Best mixed logit model: {best_mixed_logit_model}")
    print(f"Best subheading: {best_subheading}")
    print(f"Best specification: {best_specification}")

    if best_mixed_logit_model is not None:
        attribute_based_mixed_dict["Benchmarks"]["best_pca"] = best_rmse_dict[
            best_subheading
        ][best_mixed_logit_model]

    return best_aic_dict, best_rmse_dict, attribute_based_mixed_dict


###############################################################################
# 4) Figure generator
###############################################################################
def generate_figure(metric, models_dict, title, x_label, output_filename):
    """
    Plots the given `metric` (one of:
        'first_choice_ll', 'first_choice_aic', 'second_choice_ll', 'second_choice_rmse'
    )
    for each model in each subheading (Benchmarks, Product Images, etc.).

    Arguments:
    - metric: str, name of the key to plot on the X-axis
    - models_dict: dict of subheadings -> (dict of model_name -> dict of metrics)
    - title: Title for the figure
    - output_filename: Where to save the figure (png)
    """

    # We will build a list of (display_text, metric_value, subheading_label) to help with plotting
    plotting_data = []
    for subheading, model_dict in models_dict.items():
        # "subheading" e.g. "Benchmarks", "Product Titles"
        # "model_dict" is { model_name: { 'first_choice_ll':..., 'first_choice_aic':..., 'second_choice_ll':..., 'second_choice_rmse':... }, ... }
        for model_name, metrics in model_dict.items():
            # Check if we have a nicer name in `formatted_submodel_map`
            if model_name in formatted_submodel_map:
                disp_name = formatted_submodel_map[model_name]
            else:
                # fallback to model_name
                disp_name = model_name

            # The x-value for the plot
            x_value = metrics[metric]
            # Save
            plotting_data.append((subheading, disp_name, x_value))

    # We will create a list of subheading breaks
    subheading_order = [
        "Benchmarks",
        "Product Images",
        "Product Titles",
        "Product Descriptions",
        "User Reviews",
        "Attribute-Based Mixed Logit",
    ]

    # Filter only those subheadings that actually appear in the data
    subheading_order = [
        s for s in subheading_order if s in models_dict and len(models_dict[s]) > 0
    ]

    # Build a plotting order with blank lines at subheading boundaries
    final_labels = []
    final_xvalues = []
    # We'll store subheading indices to place a horizontal line or text
    subheading_indices = []
    running_index = 0

    for subh in subheading_order:
        subheading_indices.append(running_index)
        # Insert an empty row for subheading separation (unless it's the first subheading)
        final_labels.append("")  # blank label
        final_xvalues.append(None)
        running_index += 1

        # Insert each model for this subheading
        # gather the models in the order they were found in `plotting_data`
        sub_data = [d for d in plotting_data if d[0] == subh]
        for sd in sub_data:
            # sd is (subheading, disp_name, x_value)
            final_labels.append(sd[1])  # disp_name
            final_xvalues.append(sd[2])  # x_value
            running_index += 1

    # Now, let's plot them on a horizontal scatter or bar chart
    _, ax = plt.subplots(figsize=(8, 6))
    y_positions = range(len(final_labels))

    # Plot
    for i, xval in enumerate(final_xvalues):
        if xval is not None:
            ax.scatter(xval, i, color=None, alpha=1, zorder=1)
        else:
            # draw a horizontal line for subheading separation or do nothing
            ax.axhline(y=i, color="black", linestyle="--", linewidth=0.5)

    xmin, xmax = ax.get_xlim()
    xmid = (xmin + xmax) / 2.0

    for i, subh_idx in enumerate(subheading_indices):
        subh_name = subheading_order[i]
        ax.text(
            xmid,
            subh_idx,
            subh_name,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(facecolor="white", alpha=1, edgecolor="none"),
            zorder=2,
        )

    # Y ticks
    ax.set_yticks(y_positions)
    ax.set_yticklabels(final_labels, fontsize=9)

    # Add a vertical line at the benchmark value, plain logit's metric
    benchmark_value = models_dict["Benchmarks"]["plain_logit"][metric]
    ax.axvline(x=benchmark_value, color="red", linestyle="--", linewidth=0.8, zorder=-1)

    if "best_pca" in models_dict["Benchmarks"]:
        benchmark_value_2 = models_dict["Benchmarks"]["best_pca"][metric]
        ax.axvline(
            x=benchmark_value_2, color="black", linestyle="--", linewidth=0.8, zorder=-1
        )

    # Flip the y-axis so top is subheading 1
    plt.gca().invert_yaxis()

    # Do not use scientific notation
    ax.xaxis.set_major_formatter(ScalarFormatter())

    # Title and label
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)

    # Fix x-axis range and ticks for RMSE
    if metric == "second_choice_rmse":
        ax.set_xlim(0.067, 0.090)
        ax.set_xticks(
            [
                0.0675,
                0.070,
                0.0725,
                0.075,
                0.0775,
                0.080,
                0.0825,
                0.085,
                0.0875,
            ]
        )

    # Save
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved figure: {output_filename}")


def save_summary_as_xlsx(
    best_aic_dict, best_rmse_dict, attribute_based_mixed_dict, save_path
):
    """
    Saves the three dictionaries to an Excel file with three sheets:
      1) "Best AIC"
      2) "Best RMSE"
      3) "Attribute-Based"

    Columns in each sheet:
        - Model Name
        - Specification
        - First Choice LL
        - First Choice AIC
        - Second Choice LL
        - Second Choice RMSE
        - Delta RMSE
        - Percent Delta RMSE
    """

    # Helper to convert one of the dictionaries into a DataFrame
    # applying the user-friendly "Model Name" if found in formatted_submodel_map
    def dict_to_dataframe(data_dict):
        rows = []
        for subheading, model_dict in data_dict.items():
            for model_key, metrics in model_dict.items():
                if model_key in formatted_submodel_map:
                    display_name = formatted_submodel_map[model_key]
                else:
                    display_name = model_key  # fallback to the raw key

                rows.append(
                    {
                        "Model Name": f"{subheading}: {display_name}",
                        "Specification": metrics.get("specification", ""),
                        "First Choice LL": metrics.get("first_choice_ll", float("nan")),
                        "First Choice AIC": metrics.get(
                            "first_choice_aic", float("nan")
                        ),
                        "Second Choice LL": metrics.get(
                            "second_choice_ll", float("nan")
                        ),
                        "Second Choice RMSE": metrics.get(
                            "second_choice_rmse", float("nan")
                        ),
                        "Delta RMSE": metrics.get("delta_rmse", float("nan")),
                        "Percent Delta RMSE": metrics.get(
                            "percent_delta_rmse", float("nan")
                        ),
                    }
                )
        return pd.DataFrame(rows)

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        # 1) Best AIC
        df_aic = dict_to_dataframe(best_aic_dict)
        df_aic.to_excel(writer, sheet_name="Best AIC", index=False)

        # 2) Best RMSE
        df_rmse = dict_to_dataframe(best_rmse_dict)
        df_rmse.to_excel(writer, sheet_name="Best RMSE", index=False)

        # 3) Attribute-Based
        df_attr = dict_to_dataframe(attribute_based_mixed_dict)
        df_attr.to_excel(writer, sheet_name="Attribute-Based", index=False)

    print(f"Saved summary tables to: {save_path}")


###############################################################################
# 5) Main function
###############################################################################
def main():
    # Read data, build dictionaries
    # todays_date = time.strftime("%Y-%m-%d")
    todays_date = "2025-03-21"
    excel_path = f"data/experiment/output/estimation_results/mixed_logit_results_{todays_date}.xlsx"

    # extract date from the excel file name
    date = excel_path.split("_")[-1].split(".")[0]

    best_aic_dict, best_rmse_dict, attribute_based_mixed_dict = (
        read_excel_and_select_best_models(excel_path)
    )

    # Save the dictionaries to a XLSX with three sheets
    save_path = f"data/experiment/output/estimation_results/mixed_logit_summary_{todays_date}.xlsx"
    save_summary_as_xlsx(
        best_aic_dict, best_rmse_dict, attribute_based_mixed_dict, save_path
    )

    # Print all dictionaries, but only the RMSE entries
    for subheading, model_dict in attribute_based_mixed_dict.items():
        print(f"\nSubheading: {subheading}")
        for model_name, metrics in model_dict.items():
            print(f"Model: {model_name}")
            print(f"  RMSE: {metrics['second_choice_rmse']}")

    for subheading, model_dict in best_aic_dict.items():
        print(f"\nSubheading: {subheading}")
        for model_name, metrics in model_dict.items():
            print(f"Model: {model_name}", f"Specification: {metrics['specification']}")
            print(f"  RMSE: {metrics['second_choice_rmse']}")

    # Create an output directory
    out_dir = "data/experiment/output/figures"
    os.makedirs(out_dir, exist_ok=True)

    # Define a list of (metric, selection, filename_suffix)
    figure_requests = [
        ("first_choice_ll", "best_aic", "first_choice_ll_best_aic.png"),
        ("first_choice_aic", "best_aic", "first_choice_aic_best_aic.png"),
        ("second_choice_rmse", "best_aic", "second_choice_rmse_best_aic.png"),
        ("second_choice_ll", "best_aic", "second_choice_ll_best_aic.png"),
        ("first_choice_ll", "best_rmse", "first_choice_ll_best_rmse.png"),
        ("first_choice_aic", "best_rmse", "first_choice_aic_best_rmse.png"),
        ("second_choice_rmse", "best_rmse", "second_choice_rmse_best_rmse.png"),
        ("second_choice_ll", "best_rmse", "second_choice_ll_best_rmse.png"),
        ("first_choice_ll", "attribute", "first_choice_ll_attribute.png"),
        ("first_choice_aic", "attribute", "first_choice_aic_attribute.png"),
        ("second_choice_rmse", "attribute", "second_choice_rmse_attribute.png"),
        ("second_choice_ll", "attribute", "second_choice_ll_attribute.png"),
    ]

    for metric, selection, filename in figure_requests:
        if selection == "best_aic":
            dict_to_plot = best_aic_dict
        elif selection == "best_rmse":
            dict_to_plot = best_rmse_dict
        else:
            dict_to_plot = attribute_based_mixed_dict

        filename = filename.split(".")[0] + "_" + date + ".png"

        output_path = os.path.join(out_dir, filename)
        if metric == "first_choice_ll":
            fig_title = "Mixed Logit with Principal Components"
            x_label = "First Choice Log-Likelihood (LL)"
        elif metric == "first_choice_aic":
            fig_title = "Mixed Logit with Principal Components"
            x_label = "First Choice AIC"
        elif metric == "second_choice_rmse":
            fig_title = "Mixed Logit with Principal Components"
            x_label = "Counterfactual RMSE (Second Choices)"
        else:
            fig_title = "Mixed Logit with Principal Components"
            x_label = "Counterfactual Log-Likelihood (Second Choices)"
        if selection == "attribute":
            fig_title = "Attribute-Based Mixed Logit Specifications"

        generate_figure(metric, dict_to_plot, fig_title, x_label, output_path)


if __name__ == "__main__":
    main()
