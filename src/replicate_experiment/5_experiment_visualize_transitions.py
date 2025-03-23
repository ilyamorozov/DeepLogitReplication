# python -m src.replicate_experiment.5_experiment_visualize_transitions

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the Excel file
date = "2025-03-21"
file_path = f"data/experiment/output/estimation_results/mixed_logit_diversion_matrices_{date}.xlsx"
xls = pd.ExcelFile(file_path)

# Read the three sheets
review_st_df = pd.read_excel(xls, sheet_name="user_reviews_USE_text", header=None)
logit_df = pd.read_excel(xls, sheet_name="plain_logit", header=None)
raw_data_df = pd.read_excel(xls, sheet_name="data", header=None)
mlogit_df = pd.read_excel(xls, sheet_name="observables", header=None)

# Extract the book names (first row, excluding the first element)
book_names = review_st_df.iloc[0, 1:].values

# Extract matrices (columns 2-11 and rows 2-11)
pca_transitions = review_st_df.iloc[1:, 1:].values
logit_transitions = logit_df.iloc[1:, 1:].values
data_transitions = raw_data_df.iloc[1:, 1:].values
mlogit_transitions = mlogit_df.iloc[1:, 1:].values

# Load the CSV file with book genres
ebook_dict_df = pd.read_csv("data/experiment/input/books/books.csv")

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

# Define colors for plotting (Orange, Green, Blue)
colors = [
    (255 / 255, 127 / 255, 14 / 255, 1),  # Orange
    (44 / 255, 160 / 255, 44 / 255, 1),  # Green
    (31 / 255, 119 / 255, 180 / 255, 1),  # Blue
    (205 / 255, 51 / 255, 51 / 255, 1),  # Dark Red
]

# Loop through all K values (0 to 9)
for K in range(10):
    # Extract the K-th row from each transitions matrix
    pca_row = pca_transitions[K, :]
    logit_row = logit_transitions[K, :]
    data_row = data_transitions[K, :]
    mlogit_row = mlogit_transitions[K, :]

    # Assert that all rows sum to 1
    assert np.isclose(pca_row.sum(), 1)
    assert np.isclose(logit_row.sum(), 1)
    assert np.isclose(data_row.sum(), 1)
    assert np.isclose(mlogit_row.sum(), 1)

    # Organize the rows into a 10x3 array
    combined_matrix = np.column_stack((pca_row, logit_row, data_row, mlogit_row))

    # Merge with book names
    book_combined = np.column_stack((book_names_with_genres, combined_matrix))

    # Convert to a DataFrame for easier sorting
    df_combined = pd.DataFrame(
        book_combined, columns=["Book Name", "PCA", "Logit", "Data", "MixedLogit"]
    )

    # Convert the numeric columns from strings to floats
    df_combined["PCA"] = df_combined["PCA"].astype(float)
    df_combined["Logit"] = df_combined["Logit"].astype(float)
    df_combined["Data"] = df_combined["Data"].astype(float)
    df_combined["MixedLogit"] = df_combined["MixedLogit"].astype(float)

    # Sort by the "Data" column in ascending order
    df_sorted = df_combined.sort_values(by="Data", ascending=True)

    # Remove the first row
    df_sorted = df_sorted.iloc[1:]

    # Plot the resulting data (flipped, with books on Y-axis and values on X-axis)
    plt.figure(figsize=(10, 6))

    # Plot each line with markers, but now X-axis is the transition data and Y-axis is the books
    plt.plot(
        df_sorted["PCA"],
        df_sorted["Book Name"],
        label="PCA",
        marker="o",
        color=colors[0],
    )
    plt.plot(
        df_sorted["Logit"],
        df_sorted["Book Name"],
        label="Logit",
        marker="s",
        color=colors[1],
    )
    plt.plot(
        df_sorted["Data"],
        df_sorted["Book Name"],
        label="Data",
        marker="^",
        color=colors[2],
    )
    plt.plot(
        df_sorted["MixedLogit"],
        df_sorted["Book Name"],
        label="MixedLogit",
        marker="d",
        color=colors[3],
    )

    # Set the labels
    plt.xlabel("Second Choice Probability")
    plt.title(f"Second Choice Probabilities for {book_names_with_genres[K]}")

    # Remove the Y-axis label
    plt.gca().set_ylabel("")

    # Place legend in the southeast corner (lower right)
    plt.legend(loc="lower right")

    # Save the plot for each K
    output_path = f"data/experiment/output/transitions/transitions_book{K+1}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)

    # Show the plot
    # plt.show()

    # Close the plot to avoid memory issues
    plt.close()

print("All graphs saved successfully.")


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
books_csv_path = "data/experiment/input/books/books.csv"
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
# output_file = "implied_substitutes.xlsx"
output_dir = "data/experiment/output/implied_substitutes"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/implied_substitutes.xlsx"
final_table.to_excel(output_file, index=False)


# NEW
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


xlsx_path = "data/experiment/output/implied_substitutes/implied_substitutes.xlsx"
output_tex_path = "data/experiment/output/implied_substitutes/first_three_products.tex"

generate_fancy_latex(xlsx_path, output_tex_path)
