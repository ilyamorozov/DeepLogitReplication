import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "./data/experiment/output/merger_simulations/predicted_price_increases.csv"
df = pd.read_csv(file_path, delimiter=",")

# Ensure the output directory exists
output_dir = "./data/experiment/output/merger_simulations//merger_graphs/"
os.makedirs(output_dir, exist_ok=True)

# Define a function to shorten long book titles
def shorten_title(name):
    return (
        "Don't Believe" if name == "Don't Believe Everything You Think" else
        "Art of Letting Go" if name == "The Art of Letting Go" else
        "Serpent & Wings" if name == "The Serpent & The Wings of Night" else
        "Court of Ravens" if name == "Court of Ravens and Ruin" else
        "Ashes & Star" if name == "The Ashes & The Star Cursed King" else name
    )

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
    plt.scatter(x_labels_sorted, y_rel_plain_logit_sorted, marker='o', color=colors[0], label="Plain Logit")
    plt.scatter(x_labels_sorted, y_rel_mlogit_attributes_sorted, marker='s', color=colors[1], label="Mixed Logit with Attributes")
    plt.scatter(x_labels_sorted, y_rel_mlogit_texts_sorted, marker='^', color=colors[2], label="Mixed Logit with Texts")

    # Formatting
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Expected Price Increase (%)")
    plt.title(graph_title)
    plt.legend()

    # Save the figure
    plt.savefig(file_path_save, dpi=300, bbox_inches="tight")

    # Close the figure to free memory
    plt.close()
