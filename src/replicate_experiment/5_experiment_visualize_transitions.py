# python -m src.replicate_experiment.5_experiment_visualize_transitions

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.helper_functions.file_structure.get_file_path_from_config import get_file_path_from_config

def visualize_transitions(file_path, output_dir):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    # Read the three sheets
    embedding_df = pd.read_excel(xls, sheet_name="user_reviews_USE_text", header=None)
    logit_df = pd.read_excel(xls, sheet_name="plain_logit", header=None)
    raw_data_df = pd.read_excel(xls, sheet_name="data", header=None)
    mlogit_df = pd.read_excel(xls, sheet_name="observables", header=None)

    # Extract the book names (first row, excluding the first element)
    book_names = embedding_df.iloc[0, 1:].values

    # Extract matrices (columns 2-11 and rows 2-11)
    pca_transitions = embedding_df.iloc[1:, 1:].values
    logit_transitions = logit_df.iloc[1:, 1:].values
    data_transitions = raw_data_df.iloc[1:, 1:].values
    mlogit_transitions = mlogit_df.iloc[1:, 1:].values

    # Load the CSV file with book genres

    books_csv_path = get_file_path_from_config(path_type="EXPERIMENT_5", path="BOOKS_CSV_PATH")
    ebook_dict_df = pd.read_csv(books_csv_path)

    # Extract the "genre" column
    genres = ebook_dict_df["genre"].values

    # Map the genres into shorter names
    genre_mapping = {
        "Science Fiction & Fantasy": "Fantasy",
        "Mystery, Thriller & Suspense": "Mystery",
        "Self-Help": "Self-Help",
    }


    # Merge book names with their genres (combine titles and genres)
    book_names_with_genres = [
        f"{book} ({genre_mapping[genre]})" for book, genre in zip(book_names, genres)
    ]
        
    # Part 2: Create the implied substitutes table

    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    # Read the three sheets
    raw_data_df = pd.read_excel(xls, sheet_name="data", header=None)
    logit_df = pd.read_excel(xls, sheet_name="plain_logit", header=None)
    mlogit_df = pd.read_excel(xls, sheet_name="observables", header=None)
    review_st_df = pd.read_excel(xls, sheet_name="user_reviews_USE_text", header=None)

    # Extract the book names (first row, excluding the first element)
    book_names = review_st_df.iloc[0, 1:].values

    # Replace long names with shorter strings in book_names
    book_names = np.array(
        [
            (
                "Don't Believe"
                if name == "Don't Believe Everything You Think"
                else (
                    "Art of Letting Go"
                    if name == "The Art of Letting Go"
                    else (
                        "Serpent & Wings"
                        if name == "The Serpent & The Wings of Night"
                        else (
                            "Court of Ravens"
                            if name == "Court of Ravens and Ruin"
                            else (
                                "Ashes & Star"
                                if name == "The Ashes & The Star Cursed King"
                                else name
                            )
                        )
                    )
                )
            )
            for name in book_names
        ]
    )

    # Extract matrices (columns 2-11 and rows 2-11)
    data_transitions = raw_data_df.iloc[1:, 1:].values
    logit_transitions = logit_df.iloc[1:, 1:].values
    mlogit_transitions = mlogit_df.iloc[1:, 1:].values
    pca_transitions = review_st_df.iloc[1:, 1:].values

    # Load the CSV file with book genres
    ebook_dict_df = pd.read_csv(books_csv_path)

    # Extract the "genre" column
    genres = ebook_dict_df["genre"].values

    # Map the genres into shorter names
    genre_mapping = {
        "Science Fiction & Fantasy": "F",
        "Mystery, Thriller & Suspense": "M",
        "Self-Help": "S",
    }

    # Merge book names with their genres (combine titles and genres)
    book_names_with_genres = [
        f"{book} ({genre_mapping[genre]})" for book, genre in zip(book_names, genres)
    ]


    # Placeholder for all tables
    all_tables = []

    # Loop over K and perform computations
    for K in range(10):
        # Remove the K-th book and corresponding SCP values from each dataset
        book_names_sub = np.delete(book_names_with_genres, K)

        data_scp_sub = np.delete(data_transitions[K, :], K)
        logit_scp_sub = np.delete(logit_transitions[K, :], K)
        mlogit_scp_sub = np.delete(mlogit_transitions[K, :], K)
        pca_scp_sub = np.delete(pca_transitions[K, :], K)

        # Create substitutes tables
        data_transitions_substitutes = (
            pd.DataFrame({"Book": book_names_sub, "SCP": data_scp_sub})
            .sort_values(by="SCP", ascending=False)
            .reset_index(drop=True)
        )

        logit_transitions_substitutes = (
            pd.DataFrame({"Book": book_names_sub, "SCP": logit_scp_sub})
            .sort_values(by="SCP", ascending=False)
            .reset_index(drop=True)
        )

        mlogit_transitions_substitutes = (
            pd.DataFrame({"Book": book_names_sub, "SCP": mlogit_scp_sub})
            .sort_values(by="SCP", ascending=False)
            .reset_index(drop=True)
        )

        pca_transitions_substitutes = (
            pd.DataFrame({"Book": book_names_sub, "SCP": pca_scp_sub})
            .sort_values(by="SCP", ascending=False)
            .reset_index(drop=True)
        )

        # Combine all four tables side by side
        combined_table = pd.concat(
            [
                data_transitions_substitutes.rename(columns={"SCP": "Data SCP"}),
                logit_transitions_substitutes.rename(columns={"SCP": "Logit SCP"}),
                mlogit_transitions_substitutes.rename(columns={"SCP": "MLogit SCP"}),
                pca_transitions_substitutes.rename(columns={"SCP": "PCA SCP"}),
            ],
            axis=1,
        )

        # Round the SCP columns to 3 decimal places
        combined_table = combined_table.round(3)

        # Add a header to indicate the current K value based on book_names[K]
        header = pd.DataFrame(
            [[f"{book_names_with_genres[K]}", "", "", "", "", "", "", ""]],
            columns=combined_table.columns,
        )

        # Append the header and the combined table, adding a blank row at the end
        all_tables.append(header)
        all_tables.append(combined_table)
        all_tables.append(
            pd.DataFrame(
                [[""] * len(combined_table.columns)], columns=combined_table.columns
            )
        )

    # Concatenate all tables vertically
    final_table = pd.concat(all_tables, ignore_index=True)

    # Save to Excel
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/implied_substitutes.xlsx"
    final_table.to_excel(output_file, index=False)


# Table 2: first_three_products.tex
def generate_fancy_latex(xlsx_path, output_path):
    df = pd.read_excel(xlsx_path)
    df = df.applymap(lambda x: x.replace("&", r"\&") if isinstance(x, str) else x)

    # Always 3 decimals for prob columns:
    # We'll store them as strings like "0.250"
    def fmt_prob(prob):
        return f"{prob:.3f}"

    # We assume each panel block is 11 rows: [header row], [9 data rows], [blank row].
    # We'll do the first 3 panels (i=0,1,2).

    lines = []
    books_wanted = [7, 0, 5]
    for panel_letter_i, i in enumerate(books_wanted):
        block_start = i * 11
        block_end = block_start + 11
        block = df.iloc[block_start:block_end].reset_index(drop=True)

        # Row 0 is the header with the product name:
        product_name = block.iloc[0, 0]
        panel_letter = chr(ord("A") + panel_letter_i)  # A, B, C, etc.

        # Extract data rows (1..9):
        data_rows = block.iloc[1:10]

        lines.append(r"\resizebox{\textwidth}{!}{")
        lines.append(r"\centering")
        lines.append(
            rf"\textbf{{Panel {panel_letter}. Predicted Second-Choice Probabilities when First Choice is \textit{{{product_name}}}}}"
        )
        lines.append(r"}")

        # 8 centered columns
        lines.append(r"\resizebox{\textwidth}{!}{")
        lines.append(r"\begin{tabular}{c c c c c c c c}")
        lines.append(r"\toprule")

        # Bold heading row, with multi-column group headings
        lines.append(
            r"\multicolumn{2}{c}{Experimental Data} & "
            r"\multicolumn{2}{c}{Plain Logit} & "
            r"\multicolumn{2}{c}{Attribute-Based Mixed Logit} & "
            r"\multicolumn{2}{c}{Review-Based Mixed Logit} \\"
        )

        # Horizontal rule under the multi-column headings
        lines.append(
            r"\cmidrule(lr){1-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}"
        )

        # Second header row
        lines.append(
            r"Book & Prob. & " r"Book & Prob. & " r"Book & Prob. & " r"Book & Prob. \\"
        )

        lines.append(r"\midrule")

        # Now the 9 data rows:
        for row_i in range(len(data_rows)):
            row = data_rows.iloc[row_i]
            b1, p1 = row[0], fmt_prob(row[1])
            b2, p2 = row[2], fmt_prob(row[3])
            b3, p3 = row[4], fmt_prob(row[5])
            b4, p4 = row[6], fmt_prob(row[7])

            lines.append(f"{b1} & {p1} & {b2} & {p2} & {b3} & {p3} & {b4} & {p4} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\vspace{0.5cm}")  # Add some vertical space
        lines.append("")  # blank line

    # Write to a .tex file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_path} with your fancy tables.")

if __name__ == "__main__":
    date = "2025-09-15"
    # file_path = f"data/experiment/output/estimation_results/mixed_logit_diversion_matrices_{date}.xlsx"
    dir_diversion_matrices = get_file_path_from_config(path_type="EXPERIMENT_5", path="DIR_DIVERSION_MATRICES")
    diversion_matrices_path = os.path.join(dir_diversion_matrices, f"mixed_logit_diversion_matrices_{date}.xlsx")


    output_dir = get_file_path_from_config(path_type="EXPERIMENT_5", path="OUTPUT_DIR")

    visualize_transitions(diversion_matrices_path, output_dir)

    xlsx_path = get_file_path_from_config(path_type="EXPERIMENT_5", path="XLSX_PATH")
    output_tex_path = get_file_path_from_config(path_type="EXPERIMENT_5", path="OUTPUT_TEX_PATH")

    generate_fancy_latex(xlsx_path, output_tex_path)
