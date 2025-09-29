import os
from src.helper_functions.file_structure.get_file_path_from_config import get_file_path_from_config

import pandas as pd

def main():
    results_date = "2025-09-15"
    results_dir = get_file_path_from_config(path_type="EXPERIMENT_10", path="RESULTS_DIR")
    results_path = os.path.join(results_dir, f"mixed_logit_results_{results_date}.xlsx")

    all_sheets = pd.read_excel(results_path, sheet_name=None)

    all_results_df = pd.concat([df for model_name, df in all_sheets.items() if model_name != "vgg16_image"], ignore_index=True)

    best_models_df = all_results_df[all_results_df["Best Specification"] == 1]

    algo_1_df = pd.DataFrame()

    for model_name, df in all_sheets.items():
        if model_name == "vgg16_image":
            continue
        if model_name == "observables":
            algo_1_df = pd.concat([algo_1_df, df])
            continue
        selected_model = df[df["Best Specification"] == 1]
        selected_model_step = selected_model["Step"].iloc[0]
        encountered_df = df[df["Step"] <= int(selected_model_step) + 1]
        algo_1_df = pd.concat([algo_1_df, encountered_df])


    
    

    corr_all = all_results_df[["First Choice AIC", "Second Choice RMSE"]].corr()
    corr_algo_1 = algo_1_df[["First Choice AIC", "Second Choice RMSE"]].corr()
    corr_best = best_models_df[["First Choice AIC", "Second Choice RMSE"]].corr()
    
    num_rows_all = len(all_results_df)
    num_rows_algo_1 = len(algo_1_df)
    num_rows_best = len(best_models_df)

    print(num_rows_all)
    print(num_rows_algo_1)
    print(num_rows_best)
   

    print(corr_all)
    print(corr_algo_1)
    print(corr_best)

    save_path = get_file_path_from_config(path_type="EXPERIMENT_10", path="SAVE_PATH")
    with open(save_path, "w") as f:
        f.write("=== Summary ===\n")
        f.write(f"All Results: {num_rows_all} rows\n")
        f.write(f"Algo 1 Models: {num_rows_algo_1} rows\n\n")
        f.write(f"Best Models: {num_rows_best} rows\n\n")

        f.write("=== Correlation (All Results) ===\n")
        f.write(corr_all.to_string())
        f.write("\n\n")

        f.write("=== Correlation (Algorithm 1) ===\n")
        f.write(corr_algo_1.to_string())
        f.write("\n\n")

        f.write("=== Correlation (Best Models) ===\n")
        f.write(corr_best.to_string())
        f.write("\n")

    print(f"Saved correlation summary to: {save_path}")



if __name__ == "__main__":
    main()