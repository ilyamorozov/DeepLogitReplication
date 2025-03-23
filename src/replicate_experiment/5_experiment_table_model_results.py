# python -m src.replicate_experiment.6_experiment_table_model_results

import pandas as pd


def generate_mixed_logit_table(xlsx_path, output_tex):
    """
    Read 'Best AIC' and 'Attribute-Based' sheets from the Excel file,
    then produce one LaTeX table with Panel A/B/C sub-sections.
    """
    # 1. Read the two sheets
    best_aic = pd.read_excel(xlsx_path, sheet_name="Best AIC")
    attrib_based = pd.read_excel(xlsx_path, sheet_name="Attribute-Based")
    best_aic = best_aic.applymap(
        lambda x: x.replace("&", r"\&") if isinstance(x, str) else x
    )
    attrib_based = attrib_based.applymap(
        lambda x: x.replace("&", r"\&") if isinstance(x, str) else x
    )

    # 2. Rename columns for easier reference
    col_map = {
        "Model Name": "Model",
        "First Choice LL": "logL",
        "First Choice AIC": "AIC",
        "Second Choice RMSE": "RMSE",
        "Delta RMSE": "Delta",
        "Percent Delta RMSE": "Percent",
    }
    best_aic.rename(columns=col_map, inplace=True)
    attrib_based.rename(columns=col_map, inplace=True)

    # Plain logit is the benchmark model
    panel_a = attrib_based[attrib_based["Model"] == "Benchmarks: Plain Logit"].copy()
    # Everything in Best AIC goes to Panel B except Plain Logit
    panel_b = best_aic[best_aic["Model"] != "Benchmarks: Plain Logit"].copy()
    panel_b = panel_b[panel_b["Model"] != "Other: Combined"]
    # Everything in Attribute-Based goes to Panel C except Plain Logit
    panel_c = attrib_based[attrib_based["Model"] != "Benchmarks: Plain Logit"].copy()
    panel_c = panel_c[panel_c["Model"] != "Benchmarks: Best Mixed Logit (Texts)"]

    # 4. Helper to format numeric columns
    #    Here we use 3 decimals for everything except we'll handle
    #    Percent with one decimal plus a trailing '%', if you prefer.
    def fmt_num(x, decimals=3):
        if pd.isnull(x):
            return ""
        if decimals == 1:
            return f"{x:.1f}"
        return f"{x:.3f}"

    def fmt_percent(x):
        if pd.isnull(x):
            return ""
        return f"{x*100:.1f}\\%"  # e.g. -9.8 => "-9.8\%"

    def fmt_model_name(name):
        if name.startswith("Benchmarks: "):
            return name.replace("Benchmarks: ", "")
        if name.startswith("Attribute-Based Mixed Logit: "):
            if name.endswith("All Attributes"):
                return "Price, Pages, Year, \\& Genre (All Attr.)"
            return name.replace("Attribute-Based Mixed Logit: ", "")
        if name.startswith("User Reviews: "):
            return name.replace("User Reviews: ", "Reviews: ")
        if name.startswith("Product Images: "):
            return name.replace("Product Images: ", "Images: ")
        if name.startswith("Product Descriptions: "):
            return name.replace("Product Descriptions: ", "Descriptions: ")
        return name

    # 5. Build the LaTeX lines
    lines = []
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\centering")

    # 6. Begin the tabular with 6 columns:
    #    1) Model
    #    2) logL
    #    3) AIC
    #    4) RMSE
    #    5) Δ
    #    6) %
    # We group columns (2,3) under "First Choices (Data)" and (4,5,6) under "Second Choices (Counterfactual)"
    lines.append(r"\begingroup")
    lines.append(r"\setlength{\tabcolsep}{12pt}")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{2}{c}{First Choices (Data)} "
        r"& \multicolumn{3}{c}{Second Choices (Counterf.)} \\"
    )
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-6}")
    lines.append(
        r"Model & $\log L$ & \textit{AIC} & \textit{RMSE} & \multicolumn{2}{c}{Rel. to Plain Logit} \\"
    )
    lines.append(r" && & & $\Delta$ & \% \\")
    lines.append(r"\midrule")

    # 7. Panel A
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{6}{l}{\textbf{\textit{Panel A. Benchmark Model}}} \\")
    lines.append(r"\addlinespace")
    for _, row in panel_a.iterrows():
        model = fmt_model_name(row["Model"])
        logl = fmt_num(row["logL"], decimals=1)
        aic = fmt_num(row["AIC"], decimals=1)
        rmse = fmt_num(row["RMSE"])
        delta = ""
        pct = ""

        lines.append(f"{model} & {logl} & {aic} & {rmse} & {delta} & {pct} \\\\")

    # 8. Panel B
    lines.append(r"\addlinespace")
    lines.append(
        r"\multicolumn{6}{l}{\textbf{\textit{Panel B. Mixed Logit Models with Principal Components}}} \\"
    )
    lines.append(r"\addlinespace")
    for _, row in panel_b.iterrows():
        model = fmt_model_name(row["Model"])
        logl = fmt_num(row["logL"], decimals=1)
        aic = fmt_num(row["AIC"], decimals=1)
        rmse = fmt_num(row["RMSE"])
        delta = fmt_num(row["Delta"])
        pct = fmt_percent(row["Percent"])

        if model == "Reviews: Sentence Encoder (USE)":
            # underline this row
            lines.append(
                f"\\underline{{{model}}} & \\underline{{{logl}}} & \\underline{{{aic}}} & \\underline{{{rmse}}} & \\underline{{{delta}}} & \\underline{{{pct}}} \\\\"
            )
        else:
            lines.append(f"{model} & {logl} & {aic} & {rmse} & {delta} & {pct} \\\\")

    # 9. Panel C
    lines.append(r"\addlinespace")
    lines.append(
        r"\multicolumn{6}{l}{\textbf{\textit{Panel C. Attribute-Based Mixed Logit Models}}} \\"
    )
    lines.append(r"\addlinespace")
    for _, row in panel_c.iterrows():
        model = fmt_model_name(row["Model"])
        logl = fmt_num(row["logL"], decimals=1)
        aic = fmt_num(row["AIC"], decimals=1)
        rmse = fmt_num(row["RMSE"])
        delta = fmt_num(row["Delta"])
        pct = fmt_percent(row["Percent"])

        lines.append(f"{model} & {logl} & {aic} & {rmse} & {delta} & {pct} \\\\")

    # 10. Finish table
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\endgroup")
    lines.append(r"}")

    # 11. Write out
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Successfully wrote single-table LaTeX with Panels A/B/C to '{output_tex}'")


if __name__ == "__main__":

    date = "2025-03-21"
    input_xlsx = (
        f"data/experiment/output/estimation_results/mixed_logit_summary_{date}.xlsx"
    )
    output_tex = (
        f"data/experiment/output/estimation_results/model_validation_results_{date}.tex"
    )

    generate_mixed_logit_table(input_xlsx, output_tex)
