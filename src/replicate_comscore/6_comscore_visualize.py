

import os
import re
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.helper_functions.file_structure.get_file_path_from_config import get_file_path_from_config

# Function to recursively find all .xlsx files matching the pattern
def find_files(pattern, root):
    return [str(path) for path in Path(root).rglob(pattern)]


output_path = get_file_path_from_config(path_type="COMSCORE_7", path="OUTPUT_PATH")
figure_path = os.path.join(output_path, "figures")
table_path = os.path.join(output_path, "tables")
summary_path = os.path.join(output_path, "estimation_summaries")

os.makedirs(figure_path, exist_ok=True)
os.makedirs(table_path, exist_ok=True)
os.makedirs(summary_path, exist_ok=True)


# ------ SCRIPT 1 ------
# Compute and save three tables: final_table, collapsed_table, wide_table

# Set the root directory to search
root_directory = figure_path = os.path.join(output_path, "estimation_results")

xlsx_pattern = "mixed_logit_results_*.xlsx"

# Find all files whose name starts with "mlogit_pca_" and ends with ".xlsx"
file_paths = find_files(xlsx_pattern, root_directory)

dates = ["2025-08-21"]

# Filter out file paths that do not have the date in them
file_paths = [path for path in file_paths if any(date in path for date in dates)]

# Check if any files are found
if not file_paths:
    raise FileNotFoundError(
        f"No files matching the pattern f{xlsx_pattern} were found in the directory {root_directory}."
    )
print(f"Found {len(file_paths)} files matching the pattern '{xlsx_pattern}':")

# Initialize an empty list to store DataFrames from all files
all_dataframes = []

# Process each file
for file_path in file_paths:
    # Extract the numbers from the file name
    category_code = re.search(r"mixed_logit_results_(\d+)", file_path).group(1)
    print(f"Processing file '{file_path}' for category code '{category_code}'.")

    # Read all sheets into a dictionary of DataFrames
    sheets = pd.read_excel(file_path, sheet_name=None)

    # Initialize a list to store DataFrames from each file
    dataframes = []

    # Loop through each sheet and process the data
    for sheet_name, df in sheets.items():
        # Add a new column with the sheet name
        df["Embedding_Model"] = sheet_name

        # Add a new column "Data_Type" based on the Specification value
        if sheet_name == "combined":
            df["Data_Type"] = "combined"
        elif sheet_name.startswith("description"):
            df["Data_Type"] = "descriptions"
        elif sheet_name.startswith("title"):
            df["Data_Type"] = "titles"
        elif sheet_name.startswith("user_reviews"):
            df["Data_Type"] = "reviews"
        elif sheet_name.endswith("image"):
            df["Data_Type"] = "images"
        elif sheet_name.startswith("observables"):
            df["Data_Type"] = "observables"
        else:
            raise ValueError(f"Unknown data type for sheet '{sheet_name}'.")

        # Add the Category_Code column
        df["Category_Code"] = category_code

        # Append the DataFrame to the list
        dataframes.append(df)

    # Concatenate all DataFrames from this file and append to the main list
    all_dataframes.extend(dataframes)

# Concatenate all DataFrames from all files into one
final_table = pd.concat(all_dataframes, ignore_index=True)
print(f"Total unique Category_Codes in final_table: {final_table['Category_Code'].nunique()}")

# Rename First Choice AIC column to AIC, and Specification to Model
final_table.rename(
    columns={"First Choice AIC": "AIC", "Specification": "Model"}, inplace=True
)

# Ensure AIC column exists and compute AIC_min, Delta, ExpDelta, and Akaike_Weight
if "AIC" not in final_table.columns:
    raise ValueError("Column 'AIC' not found in the dataset.")

# Group by Category_Code to perform within-group computations
final_table["AIC_min"] = final_table.groupby("Category_Code")["AIC"].transform("min")
final_table["Delta"] = final_table["AIC"] - final_table["AIC_min"]

# Assert that Delta is non-negative
assert (
    final_table["Delta"] >= 0
).all(), "Some Delta values are negative, which is unexpected."

# Compute ExpDelta and Akaike_Weight
final_table["ExpDelta"] = np.exp(-final_table["Delta"] / 2)
final_table["Akaike_Weight"] = final_table.groupby("Category_Code")[
    "ExpDelta"
].transform(lambda x: x / x.sum())

# Update Data_Type for rows where Model starts with "logit"
final_table.loc[final_table["Model"].str.startswith("plain"), "Data_Type"] = "logit"
print(final_table["Model"].value_counts())
print(final_table["Data_Type"].value_counts())

# Import category_list_batching.xlsx
category_list_batching_path = (os.path.join(output_path, "product_lists/category_list_batching.xlsx"))
product_list = pd.read_excel(category_list_batching_path)

# Clean up 'cat_code4' and ensure all strings are consistent
product_list["cat_code4"] = product_list["cat_code4"].astype(str).str.strip()
product_list = product_list.apply(
    lambda col: col.str.strip() if col.dtype == "object" else col
)

# Ensure Category_Code in final_table is also a clean string
final_table["Category_Code"] = final_table["Category_Code"].astype(str).str.strip()

# Merge additional columns into final_table based on Category_Code
final_table = final_table.merge(
    product_list, left_on="Category_Code", right_on="cat_code4", how="left"
)

# Fill in specific category values manually
manual_category_mapping = {
    "13060114": {
        "category1": "Electronics",
        "category2": "Computers & Accessories",
        "category3": "Computer Accessories & Peripherals",
        "category4": "Memory Cards",
        "mean_price": 12.26,
        "num_transactions": 280,
    },
    "13060404": {
        "category1": "Electronics",
        "category2": "Computers & Accessories",
        "category3": "Computers & Tablets",
        "category4": "Tablets",
        "mean_price": 149.31,
        "num_transactions": 593,
    },
    "13060701": {
        "category1": "Electronics",
        "category2": "Computers & Accessories",
        "category3": "Monitors",
        "category4": "Monitors",
        "mean_price": 127.00,
        "num_transactions": 278,
    },
    "13110101": {
        "category1": "Electronics",
        "category2": "Headphones",
        "category3": "Earbud Headphones",
        "category4": "Earbud Headphones",
        "mean_price": 81.51,
        "num_transactions": 1598,
    },
}

for code, categories in manual_category_mapping.items():
    for col, value in categories.items():
        final_table.loc[final_table["Category_Code"] == code, col] = value

# Drop unnecessary columns after merging
columns_to_drop = ["cat_code1", "cat_code2", "cat_code3", "asin_code"]
final_table = final_table.drop(columns=columns_to_drop, errors="ignore")

# Create the second table by collapsing the data
collapsed_table = final_table.groupby(
    ["Category_Code", "Data_Type"], as_index=False
).agg(
    Total_Akaike_Weight=("Akaike_Weight", "sum"),
    category1=("category1", "first"),
    category2=("category2", "first"),
    category3=("category3", "first"),
    category4=("category4", "first"),
    num_transactions=("num_transactions", "first"),
    mean_price=("mean_price", "first"),
)

# Format Total_Akaike_Weight to three decimal points
collapsed_table["Total_Akaike_Weight"] = collapsed_table["Total_Akaike_Weight"].map(
    lambda x: f"{x:.3f}"
)

# Create the third table by pivoting the collapsed table
wide_table = collapsed_table.pivot(
    index="Category_Code", columns="Data_Type", values="Total_Akaike_Weight"
).reset_index()

# Merge additional category details to wide_table
wide_table = wide_table.merge(
    collapsed_table[
        [
            "Category_Code",
            "category1",
            "category2",
            "category3",
            "category4",
            "num_transactions",
            "mean_price",
        ]
    ].drop_duplicates(),
    on="Category_Code",
    how="left",
)

# Add the "sum_check" column to verify row sums
column_order = ["images", "titles", "descriptions", "reviews", "logit", "observables"]
# Fill missing values of observables with 0
wide_table["observables"] = wide_table["observables"].fillna(0)
print(wide_table.info())
wide_table[column_order] = wide_table[column_order].astype(float, errors="ignore")
wide_table["sum_check"] = wide_table[column_order].sum(axis=1)


# Compute average values for numeric columns
average_row = wide_table[column_order + ["sum_check"]].mean()

# Add empty placeholders for non-numeric columns in the average row
average_row_data = (
    ["Average"]
    + list(average_row)
    + [
        ""
        for _ in [
            "category1",
            "category2",
            "category3",
            "category4",
            "num_transactions",
            "mean_price",
        ]
    ]
)
wide_table.loc[len(wide_table)] = average_row_data

os.makedirs(summary_path, exist_ok=True)
final_table.to_csv(f"{summary_path}/final_table_2025-08-21.csv", index=False)
collapsed_table.to_csv(f"{summary_path}/collapsed_table_2025-08-21.csv", index=False)
wide_table.to_csv(f"{summary_path}/wide_table_2025-08-21.csv", index=False)

print(f"1. Final table: '{summary_path}/final_table_2025-08-21.csv'")
print(f"2. Collapsed table: '{summary_path}/collapsed_table_2025-08-21.csv'")
print(f"3. Wide table: '{summary_path}/wide_table_2025-08-21.csv'")

# Summary statistics across categories
columns_of_interest = ["num_transactions", "mean_price"]
wide_table[columns_of_interest] = wide_table[columns_of_interest].apply(
    pd.to_numeric, errors="coerce"
)
summary_stats = wide_table[columns_of_interest].agg(["mean", "min", "max"])
percentiles = wide_table[columns_of_interest].quantile([0.25, 0.50, 0.75]).T
percentiles.columns = ["25%", "50%", "75%"]
summary_stats = pd.concat([summary_stats, percentiles], axis=1)
print(summary_stats)


# ------ SCRIPT 2 ------
# AIC histogram for all categories

# Define file paths
file_name = "final_table_2025-08-21.csv"
file_path = f"{summary_path}/{file_name}"

category_labels_path = os.path.join(output_path, "product_lists", "category_labels_short.csv")

# Load the main and category labels tables into DataFrames
table_labeled = pd.read_csv(file_path)
category_labels = pd.read_csv(category_labels_path)

# Capitalize the first letter of strings in Data_Type
table_labeled["Data_Type"] = table_labeled["Data_Type"].str.capitalize()

# Function to assign "Model_Type" based on "Specification" and "Data_Type"
def assign_model_type(row):
    if row["Data_Type"] == "Logit":
        return "Plain Logit"
    spec = row["Embedding_Model"]
    if "COUNT" in spec:
        return "COUNT"
    elif "TFIDF" in spec:
        return "TFIDF"
    elif "USE" in spec:
        return "USE"
    elif "ST" in spec:
        return "ST"
    elif "inceptionv3" in spec:
        return "Inceptionv3"
    elif "resnet50" in spec:
        return "Resnet50"
    elif "vgg16" in spec:
        return "VGG16"
    elif "vgg19" in spec:
        return "VGG19"
    elif "xception" in spec:
        return "Xception"
    elif "combined" in spec:
        return "All Models"
    return None


# Apply the function to create the "Model_Type" column
table_labeled["Model_Type"] = table_labeled.apply(assign_model_type, axis=1)

# (Temporary) Drop models that combine both text and image embeddings
table_labeled = table_labeled[table_labeled["Model_Type"] != "All Models"]

# Collapse table_labeled by keeping the row with the greatest Delta within each Category_Code
idx_max_div = table_labeled.groupby("Category_Code")["Delta"].idxmin()
table_labeled = table_labeled.loc[idx_max_div]

diversion_summary_path = os.path.join(output_path, "elasticities", "diversions_summary.csv")
diversion_summary = pd.read_csv(diversion_summary_path)
diversion_summary.rename(columns={"Unnamed: 0": "Category_Code"}, inplace=True)

# Calculate the differences
diversion_summary["div_diff"] = diversion_summary["PCA Average of diversion ratios to closest substitute"] - diversion_summary["Plain Logit Average of diversion ratios to closest substitute"]
diversion_summary["AIC_diff"] = diversion_summary["PCA AIC"] - diversion_summary["Plain Logit AIC"] 

# Select and reorder columns from table_labeled
columns_to_keep = [
    "Category_Code",
    "category1",
    "category2",
    "category3",
    "category4",
    "Data_Type",
    "Model_Type",
]
table_labeled = table_labeled[columns_to_keep]

# Merge table_labeled with diversions_summary on Category_Code
table_category_level = table_labeled.merge(diversion_summary, on="Category_Code", how="left")

data = table_category_level["AIC_diff"]

# Create KDE object
kde = gaussian_kde(data)

# Adjust bandwidth manually
bandwidth = 4  # Specify the desired bandwidth
kde.set_bandwidth(
    bw_method=bandwidth / np.std(data)
)  # Normalize by the data's standard deviation

# Define range for KDE evaluation
x_range = np.linspace(-90, 0, 500)
y_values = kde(x_range)

# Calculate the average value of AIC_diff
avg_aic_diff = round(data.mean(), 1)

# Plot the KDE and histogram
plt.figure(figsize=(8, 6))  # Adjusted to make the plot narrower

# Plot histogram with 40 bins
plt.hist(
    data,
    bins=40,
    range=(-90, 0),
    density=True,
    alpha=0.8,
    color="#efd583",  # Interior color of histogram bars
    edgecolor="black",  # Border color of histogram bars
)

# Plot the KDE
plt.plot(
    x_range,
    y_values,
    color="#e0ab08",  # Line color for the density estimate
    linewidth=2,
    alpha=0.9,
)

# Ensure the minimum Y-value is 0
plt.ylim(bottom=0)

# Set X-axis ticks
plt.xticks(np.arange(-90, 1, 10))

# Add titles and labels
plt.title(
    f"$\\Delta$ AIC Improvement Across {len(table_category_level)} Categories (Average $\\Delta$AIC={avg_aic_diff})"
)
plt.xlabel("$\\Delta$ AIC")
plt.ylabel("Density")

# Save the plot
kde_path = f"{figure_path}/aic_histogram.png"
plt.savefig(kde_path, dpi=300)

# ------ SCRIPT 3 ------
# LaTeX table all categories

# Merge with category labels to add Category_Label_Short
table_category_level = table_category_level.merge(
    category_labels[["Category_Code", "Category_Label_Short"]],
    on="Category_Code",
    how="left",
)

selected_columns = [
    "Category_Label_Short",
    "Data_Type",
    "Model_Type",
    "AIC_diff",
    "div_diff"
]
df_selected = table_category_level[selected_columns]

# Numbered category labels
df_selected["Category_Label_Short"] = [
    f"{i+1}. {category}" for i, category in enumerate(df_selected["Category_Label_Short"])
]

# Rename columns
df_selected.columns = [
    "Category",
    "Data Type",
    "Model Type",
    "$\\Delta$ AIC",
    "$\\Delta$ Diversion to Closest Substitute",
]
df_selected.loc[:, "Category"] = df_selected["Category"].str.replace("&", r"\&")

# Compute averages 
avg_aic = pd.to_numeric(df_selected["$\\Delta$ AIC"], errors="coerce").mean()
avg_div = pd.to_numeric(df_selected["$\\Delta$ Diversion to Closest Substitute"], errors="coerce").mean()  # still in [0,1] terms

# Format diversion column as percentage strings
df_selected["$\\Delta$ Diversion to Closest Substitute"] = df_selected["$\\Delta$ Diversion to Closest Substitute"].apply(lambda x: f"{x * 100:.1f}\\%")

column_format = r"p{5cm}>{\centering\arraybackslash}p{2cm}>{\centering\arraybackslash}p{2cm}>{\centering\arraybackslash}p{1.5cm}>{\centering\arraybackslash}p{3cm}"

# Render LaTeX
latex_table = df_selected.to_latex(
    buf=None,
    float_format="{:.1f}".format,
    index=False,
    escape=False,             
    column_format=column_format,
).strip()

# Add a \footnotesize line in the very beginning
footnotesize_line = r"\{footnotesize}"
latex_table = latex_table.replace(r"\begin{tabular}", footnotesize_line + "\n\\begin{tabular}")

# Append a horizontal line + Averaged row just before \end{tabular}
# Order: Category, Data Type, Model Type, Δ AIC, Δ diversion%
avg_row_latex = (
    "\\hline\n"
    f"Averaged &  &  & {avg_aic:.1f} & {avg_div * 100:.1f}\\% \\\\"
)

latex_table = latex_table.replace("\\end{tabular}", avg_row_latex + "\n\\end{tabular}")

# 8) Save
with open(f"table_path/category_level_results_2025-08-21.tex", "w") as f:
    f.write(latex_table)


# ------ SCRIPT 4 ------
# Akaike weights stacked bar chart for all categories

# Define file paths
wide_table_path = f"{summary_path}/wide_table_2025-08-21.csv"
output_file = f"{summary_path}/akaike_weights_all_categories.png"

# Load the data
wide_table = pd.read_csv(wide_table_path)
category_labels = pd.read_csv(category_labels_path)

# Drop average (stored in last row)
wide_table = wide_table[wide_table["Category_Code"] != "Average"]

# Merge the tables on 'Category_Code'
wide_table["Category_Code"] = wide_table["Category_Code"].astype(str)
category_labels["Category_Code"] = category_labels["Category_Code"].astype(str)
merged_table = wide_table.merge(category_labels, on="Category_Code", how="left")
merged_table = merged_table.loc[:, ~merged_table.columns.str.endswith("_y")]

# Keep only the required columns
columns_to_keep = [
    "Category_Label_Short",
    "descriptions",
    "images",
    "reviews",
    "titles",
]
filtered_table = merged_table[columns_to_keep]

# Rescale numeric columns (descriptions, images, reviews, titles, logit) to sum to 1 in each row
numeric_columns = ["descriptions", "images", "reviews", "titles"]
filtered_table.loc[:, numeric_columns] = filtered_table[numeric_columns].div(
    filtered_table[numeric_columns].sum(axis=1), axis=0
)

# Set 'Category_Label_Short' as the index for plotting
filtered_table.set_index("Category_Label_Short", inplace=True)
filtered_table.loc[:, "reviews"] = filtered_table["reviews"].fillna(0)

# Sort the table by the 'images' column in ascending order
filtered_table = filtered_table.sort_values(by="images", ascending=True)

# Reorder the columns for the desired stacking order
filtered_table = filtered_table[["images", "titles", "descriptions", "reviews"]]

# Define custom colors
colors = [
    "#fc8d55",
    "#fee08a",
    "#ffffbf",
    "#e6f598",
]  # Colors for 'images', 'titles', 'descriptions', 'reviews'

# Adjust hatching line thickness
mpl.rcParams["hatch.linewidth"] = 0.1  

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(
    axis="y", labelsize=8
)  
ax.margins(y=0)  # Removes extra white space around the bars
ax.set_xlim(0, 1.0)  # Set the x-axis limits to exactly 0 and 1.0

# Plot each column
for i, column in enumerate(filtered_table.columns):
    bar = ax.barh(
        filtered_table.index,
        filtered_table[column],
        left=filtered_table.iloc[:, :i].sum(axis=1) if i > 0 else 0,  # Stacked bar
        color=colors[i],
        label=column,
        edgecolor=None,  # No black edges around bars
    )

    # Apply hatching
    if column == "reviews":
        for patch in bar:
            patch.set_hatch("////")  # Hatching pattern

# Customize the plot
plt.title(f"Akaike Weights of Different Data Types Across {len(wide_table)} Categories")
plt.xlabel("Akaike Weight")
plt.ylabel("")  # Remove y-label
plt.legend(title="Data Types", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save the plot to the specified file
output_file = f"{figure_path}/akaike_weights_all_categories.png"
plt.savefig(output_file, dpi=300)


# ------ SCRIPT 5 ------
# Plots for diversion ratios to closest substitute

selected_columns = [
    "Category_Label_Short",
    "Category_Code",
    "PCA Average of diversion ratios to closest substitute",
    "Plain Logit Average of diversion ratios to closest substitute",
    "Attribute-Based PCA Average of diversion ratios to closest substitute",
    
]
df_plot = table_category_level[selected_columns]

df_plot.set_index("Category_Label_Short", inplace=True)

# Rename columns for clarity
df_plot.rename(
    columns={
        "PCA Average of diversion ratios to closest substitute": "unstructured_data",
        "Plain Logit Average of diversion ratios to closest substitute": "plain_logit",
        "Attribute-Based PCA Average of diversion ratios to closest substitute": "attr_based",
    },
    inplace=True,
)
# Ensure numeric and convert to %
for col in ["plain_logit", "unstructured_data", "attr_based"]:
    if col not in df_plot.columns:
        df_plot[col] = np.nan
df_plot[["plain_logit", "unstructured_data", "attr_based"]] = \
    df_plot[["plain_logit", "unstructured_data", "attr_based"]].astype(float) * 100.0

# Build desired order: (1) main cats on top (A→Z), (2) others A→Z below
MAIN_CATS = {"13060404", "13060114", "13060701", "13110101"}
is_main = df_plot["Category_Code"].isin(MAIN_CATS)
labels_main = sorted(df_plot[is_main].index.tolist())
labels_rest = sorted(df_plot[~is_main].index.tolist())
top_to_bottom = labels_main + labels_rest

def order_for_barh_top_to_bottom(df, labels_top_to_bottom):
    """
    barh draws the LAST index at the TOP. To have top_to_bottom visually,
    reverse the index order after selecting.
    """
    existing = [lab for lab in labels_top_to_bottom if lab in df.index]
    return df.loc[existing[::-1]]

# Common plotting style
mpl.rcParams["hatch.linewidth"] = 0.2
mpl.rcParams["hatch.color"] = "black"  

light_blue = "#c6dbef"   # Our Approach
mid_blue   = "#6baed6"   # Attr-based
dark_blue  = "#084594"   # Plain Logit

def _final_touches(ax, title, nrows, xmin=-1):
    # Y label size & grid
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="both")

    # Margins
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.1, ymax + 0.1)

    xmin_current, xmax_current = ax.get_xlim()
    ax.set_xlim(xmin, xmax_current * 1.05)

    ax.set_xlabel("Predicted Diversion (%)")
    ax.set_ylabel("")
    ax.set_title(title)

# 1) Plot ONLY PLAIN LOGIT ---
def plot_plain_only(df, output_path=None):
    plot_df = order_for_barh_top_to_bottom(df[["plain_logit", "Category_Code"]], top_to_bottom)
    vals = plot_df["plain_logit"].values
    labels = plot_df.index

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(labels, vals, height=0.65, color=dark_blue, edgecolor="black", linewidth=0.6, label="Plain Logit")

    _final_touches(ax, f"Predicted Diversion to Closest Substitute Across {len(df)} Categories", len(df))
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

# 2) Plot PLAIN LOGIT + OUR APPROACH for ALL categories (overlapped) ---
def plot_plain_vs_unstructured(df, output_path=None):
    plot_df = order_for_barh_top_to_bottom(df[["plain_logit", "unstructured_data", "Category_Code"]], top_to_bottom)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Background (Our Approach) hatched
    ax.barh(plot_df.index, plot_df["unstructured_data"].values,
            height=0.65, color=light_blue, edgecolor="black", linewidth=0.6, hatch="////",
            label="Our Approach")
    # Foreground (Plain Logit)
    ax.barh(plot_df.index, plot_df["plain_logit"].values,
            height=0.65, color=dark_blue, edgecolor="black", linewidth=0.6,
            label="Plain Logit")

    _final_touches(ax, f"Predicted Diversion to Closest Substitute Across {len(df)} Categories", len(df))
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()



# 3) Plot PLAIN + OUR for non-main; PLAIN + OUR + ATTR for main (layered, value-aware order) ---
def plot_with_attr_for_main(df, output_path=None):
    cols = ["plain_logit", "unstructured_data", "attr_based", "Category_Code"]
    plot_df = order_for_barh_top_to_bottom(df[cols], top_to_bottom)

    idx = plot_df.index
    our = plot_df["unstructured_data"]
    plain = plot_df["plain_logit"]
    # Only draw attr for MAIN_CATS; NaN for others means nothing is drawn
    attr = plot_df["attr_based"].where(plot_df["Category_Code"].isin(MAIN_CATS), np.nan)

    # Determine which series should be in the BACK (per-row)
    # True where attr >= our (including equal); False otherwise (or NaN → False)
    mask_attr_back = (attr >= our).fillna(False)
    mask_our_back = ~mask_attr_back

    fig, ax = plt.subplots(figsize=(10, 6))

    # Grid behind bars
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="both", zorder=0)

    # Where OUR should be the background, ATTR overlays
    ax.barh(idx[mask_our_back], our[mask_our_back].values,
            height=0.65, color=light_blue, edgecolor="black", linewidth=0.6, hatch="////",
            label="Our Approach", zorder=2)
    ax.barh(idx[mask_our_back], attr[mask_our_back].values,
            height=0.65, color=mid_blue, edgecolor="black", linewidth=0.6, 
            # no label here to avoid duplicate legend
            zorder=3)

    # Where ATTR should be the background, OUR overlays 
    ax.barh(idx[mask_attr_back], attr[mask_attr_back].values,
            height=0.65, color=mid_blue, edgecolor="black", linewidth=0.6,
            label="Standard Attributes", zorder=2)
    ax.barh(idx[mask_attr_back], our[mask_attr_back].values,
            height=0.65, color=light_blue, edgecolor="black", linewidth=0.6, hatch="////",
            zorder=3)

    # Plain Logit always on top (narrowest)
    ax.barh(idx, plain.values,
            height=0.55, color=dark_blue, edgecolor="black", linewidth=0.6,
            label="Plain Logit", zorder=4)

    # Cosmetics: margins, labels, legend outside
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(-1, xmax * 1.05)

    ax.set_xlabel("Predicted Diversion (%)")
    ax.set_ylabel("")
    ax.set_title(f"Predicted Diversion to Closest Substitute Across {len(df)} Categories")

    # Deduplicate legend handles (since some labels plotted twice)
    handles, labels = ax.get_legend_handles_labels()
    seen, handles_dedup, labels_dedup = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "":
            handles_dedup.append(h)
            labels_dedup.append(l)
            seen.add(l)
    ax.legend(handles_dedup, labels_dedup, loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

plot_plain_only(df_plot, output_path=os.path.join(figure_path, "diversion_plain_only.png"))
plot_plain_vs_unstructured(df_plot, output_path=os.path.join(figure_path, "diversion_plain_vs_our.png"))
plot_with_attr_for_main(df_plot, output_path=os.path.join(figure_path, "diversion_plain_vs_our_vs_attr.png"))

# ------ SCRIPT 6 ------
# LaTeX table for AIC in main electronic categories

df = (
    table_category_level[
        table_category_level["Category_Code"].astype(str).isin(MAIN_CATS)
    ][["Category_Label_Short", "PCA AIC", "Plain Logit AIC", "Attribute-Based PCA AIC"]]
    .copy()
)

order = [
    "Electronics Tablets",
    "Electronics Monitors",
    "Electronics Memory Cards",
    "Electronics Headphones",
]
df["__ord"] = pd.Categorical(df["Category_Label_Short"], categories=order, ordered=True)
df = df.sort_values("__ord").drop(columns="__ord")

df["Category_Label_Short"] = df["Category_Label_Short"].str.replace(
    "Electronics ", "Category: ", regex=False
)

def f1(x):  # one-decimal formatting
    x = pd.to_numeric(x, errors="coerce")
    return "" if pd.isna(x) else f"{x:.1f}"

lines = []
lines.append(r"\begin{centering}")
lines.append(r"\begin{tabular}{>{\raggedright\arraybackslash}p{6cm}>{\centering\arraybackslash}p{2cm}>{\centering\arraybackslash}p{4cm}}")
lines.append(r"\hline")
lines.append(r"& {\footnotesize{$AIC$}} & {\footnotesize{$\Delta AIC$}\textbf{ Relative to Plain Logit}}\tabularnewline")
lines.append(r"\hline")

for _, r in df.iterrows():
    cat = r["Category_Label_Short"]
    aic_plain = pd.to_numeric(r["Plain Logit AIC"], errors="coerce")
    aic_attr = pd.to_numeric(r["Attribute-Based PCA AIC"], errors="coerce")
    aic_pca  = pd.to_numeric(r["PCA AIC"], errors="coerce")
    d_attr = aic_attr - aic_plain
    d_pca  = aic_pca  - aic_plain

    # Category header row
    lines.append(fr"{{\footnotesize\textbf{{{cat}}}}} &  & \tabularnewline")

    # Mixed Logit with Attributes
    lines.append(
        fr"{{\footnotesize Mixed Logit with Attributes}} & "
        fr"{{\footnotesize {f1(aic_attr)}}} & "
        fr"{{\footnotesize {f1(d_attr)}}}\tabularnewline"
    )

    # Mixed Logit with Unstructured Data
    lines.append(
        fr"{{\footnotesize Mixed Logit with Unstructured Data}} & "
        fr"{{\footnotesize {f1(aic_pca)}}} & "
        fr"{{\footnotesize {f1(d_pca)}}}\tabularnewline"
    )
    lines.append("")

lines.append(r"\hline")
lines.append(r"\end{tabular}")
lines.append(r"\par\end{centering}")

latex_table = "\n".join(lines)

with open(f"{table_path}/main_electronic_categories_AIC.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

