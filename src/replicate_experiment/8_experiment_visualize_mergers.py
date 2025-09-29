import os
import pandas as pd
import matplotlib.pyplot as plt

from src.helper_functions.file_structure.get_file_path_from_config import get_file_path_from_config

# Define a function to shorten long book titles
def shorten_title(name):
    return (
        "Don't Believe" if name == "Don't Believe Everything You Think" else
        "Art of Letting Go" if name == "The Art of Letting Go" else
        "Serpent & Wings" if name == "The Serpent & The Wings of Night" else
        "Court of Ravens" if name == "Court of Ravens and Ruin" else
        "Ashes & Star" if name == "The Ashes & The Star Cursed King" else name
    )

def fmt_percent(x, decimals=1):
        if pd.isnull(x):
            return ""
        if decimals == 1:
            return f"{x:.1f}\\%"
        return f"{x:.3f}\\%"

def main(file_path, output_dir, tex_dir):
    # Load the data
    df = pd.read_csv(file_path, delimiter=",")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for the markers
    colors = [
        (255/255, 127/255, 14/255, 1),   # Orange
        (44/255, 160/255, 44/255, 1),    # Green
        (31/255, 119/255, 180/255, 1)    # Blue
    ]

    # Get unique values of book_i
    unique_books = df["book_i"].unique()

    # Loop over each unique book_i value
    for book_id in unique_books:
        # Filter data for the current book_i
        df_filtered = df[df["book_i"] == book_id].copy()

        # Compute the difference between text-based and plain logit relative price increases
        df_filtered["diff_text_minus_plain"] = (
            df_filtered["rel_price_increase_avg_mlogit_texts"] - df_filtered["rel_price_increase_avg_plain_logit"]
        )

        # Sort the dataframe by diff_text_minus_plain in descending order
        df_filtered_sorted = df_filtered.sort_values(by="diff_text_minus_plain", ascending=False)

        # Apply the shortening function to sorted labels
        x_labels_sorted = df_filtered_sorted["title_j"].apply(shorten_title)

        # Extract sorted y-values
        y_rel_plain_logit_sorted = df_filtered_sorted["rel_price_increase_avg_plain_logit"]
        y_rel_mlogit_attributes_sorted = df_filtered_sorted["rel_price_increase_avg_mlogit_attributes"]
        y_rel_mlogit_texts_sorted = df_filtered_sorted["rel_price_increase_avg_mlogit_texts"]

        # Extract title_i for the graph title with quotation marks
        title_i_value = df_filtered_sorted["title_i"].iloc[0]
        graph_title = f'Merger Simulation: Expected Price Increase in % ("{title_i_value}")'

        # Define the filename using book_i value
        file_name = f"merger_simulation_{book_id}.png"
        file_path_save = os.path.join(output_dir, file_name)

        # Generate and save the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x_labels_sorted, y_rel_plain_logit_sorted, marker='o', color=colors[0], label="Plain Logit", zorder=3)
        plt.scatter(x_labels_sorted, y_rel_mlogit_attributes_sorted, marker='s', color=colors[1], label="Mixed Logit with Attributes", zorder=3)
        plt.scatter(x_labels_sorted, y_rel_mlogit_texts_sorted, marker='^', color=colors[2], label="Mixed Logit with Texts", zorder=3)

        #Adding dashed redline at 5% threshold
        plt.axhline(y=5, color='red', linestyle='--', linewidth=1.5, zorder=1)

        # Formatting
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Expected Price Increase (%)")
        plt.title(graph_title)
        plt.legend()


        # Save the figure
        plt.savefig(file_path_save, dpi=300, bbox_inches="tight")

        # Close the figure to free memory
        plt.close()

        # Build latex table which presents merger simulation results for each book
        lines = []
        lines.append(r"\begin{centering}")
        lines.append(r"\begin{tabular}{>{\raggedright}p{7.5cm}>{\raggedright}p{3cm}>{\centering}p{2cm}>{\centering}p{2cm}}")
        lines.append(r"\hline")
        lines.append(r"{\footnotesize\textbf{Book}} & {\footnotesize\textbf{Plain Logit}} & {\footnotesize\textbf{Mixed Logit with Attributes}} & {\footnotesize\textbf{Mixed Logit with Texts}}\tabularnewline")
        lines.append(r"\hline")

        for book_title, plain_logit, mlogit_attributes, mlogit_texts in zip(x_labels_sorted,
                                                                            y_rel_plain_logit_sorted,
                                                                            y_rel_mlogit_attributes_sorted,
                                                                            y_rel_mlogit_texts_sorted):
            if "&" in book_title:
                book_title = book_title.replace("&", "\\&")

            line = (
                    rf"{{\footnotesize {book_title}}} & "
                    rf"{{\footnotesize {fmt_percent(plain_logit)}}} & "
                    rf"{{\footnotesize {fmt_percent(mlogit_attributes)}}} & "
                    rf"{{\footnotesize {fmt_percent(mlogit_texts)}}}\tabularnewline"
                )
            lines.append(line)

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\par\end{centering}")

        with open(f"{tex_dir}merger_simulation_{book_id}.tex", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))



if __name__ == "__main__":
    merger_sims_path = get_file_path_from_config(path_type="EXPERIMENT_8", path="MERGER_SIMS_PATH")
    merger_visualizations_output_dir = get_file_path_from_config(path_type="EXPERIMENT_8", path="MERGER_VISUALIZATIONS_OUTPUT_DIR")
    tex_dir = get_file_path_from_config(path_type="EXPERIMENT_8", path="TEX_DIR")
    main(file_path=merger_sims_path, output_dir=merger_visualizations_output_dir, tex_dir=tex_dir)