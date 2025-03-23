# python -m src.replicate_comscore.5_comscore_visualize

# SCRIPT 1

import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# Function to recursively find all .xlsx files matching the pattern
def find_files(pattern, root):
    return [str(path) for path in Path(root).rglob(pattern)]


# Set the root directory to search
root_directory = "data/comscore/output/estimation_results/"

xlsx_pattern = "mixed_logit_results_*.xlsx"

# Find all files whose name starts with "mlogit_pca_" and ends with ".xlsx"
file_paths = find_files(xlsx_pattern, root_directory)

dates = ["2025-03-19", "2025-03-20"]
# Filter out file paths that do not have the date in them
# file_paths = [path for path in file_paths if date in path]
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
category_list_batching_path = (
    "data/comscore/output/product_lists/category_list_batching.xlsx"
)
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
column_order = ["images", "titles", "descriptions", "reviews", "combined", "logit"]
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

# Save outputs
output_directory = "data/comscore/output/comscore_summary"
os.makedirs(output_directory, exist_ok=True)
final_table.to_csv(f"{output_directory}/final_table.csv", index=False)
collapsed_table.to_csv(f"{output_directory}/collapsed_table.csv", index=False)
wide_table.to_csv(f"{output_directory}/wide_table.csv", index=False)

print("All three tables have been saved with additional columns:")
print(f"1. Final table: '{output_directory}/final_table.csv'")
print(f"2. Collapsed table: '{output_directory}/collapsed_table.csv'")
print(f"3. Wide table: '{output_directory}/wide_table.csv'")

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

# SCRIPT 2

# Define file paths
file_name = "final_table.csv"
file_path = f"{output_directory}/{file_name}"

category_labels_path = "data/comscore/output/product_lists/category_labels_short.csv"

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

elasticity_summary_path = "data/comscore/output/elasticities/elasticity_summary.csv"
elasticity_summary = pd.read_csv(elasticity_summary_path)
elasticity_summary.rename(columns={"Unnamed: 0": "Category_Code"}, inplace=True)

# Merge table_labeled with elasticities_summary on Category_Code
table_labeled = table_labeled.merge(elasticity_summary, on="Category_Code", how="left")


# Create table_category_level
def process_category_level(df):
    # Get AIC_logit for each Category_Code
    aic_logit = (
        df[df["Data_Type"] == "Logit"]
        .groupby("Category_Code")["AIC"]
        .min()
        .reset_index()
        .rename(columns={"AIC": "AIC_logit"})
    )

    # Merge AIC_logit back to the original DataFrame
    df = df.merge(aic_logit, on="Category_Code", how="left")

    # Find the row with the lowest AIC within each Category_Code
    idx_min_aic = df.groupby("Category_Code")["AIC"].idxmin()
    df_filtered = df.loc[idx_min_aic]

    # Add AIC_diff column
    df_filtered["AIC_diff"] = df_filtered["AIC"] - df_filtered["AIC_logit"]

    print(df_filtered.info())

    # Add HHI_diff column
    df_filtered["HHI_diff"] = (
        df_filtered["PCA Average HHI of diversion ratios"]
        / df_filtered["Plain Logit Average HHI of diversion ratios"]
        - 1.0
    ) * 100

    # Select and reorder columns
    columns_to_keep = [
        "Category_Code",
        "category1",
        "category2",
        "category3",
        "category4",
        "Data_Type",
        "Model_Type",
        "AIC_logit",
        "AIC",
        "AIC_diff",
        "HHI_diff",
    ]
    return df_filtered[columns_to_keep]


table_category_level = process_category_level(table_labeled)

# Merge with category labels to add Category_Label_Short
table_category_level = table_category_level.merge(
    category_labels[["Category_Code", "Category_Label_Short"]],
    on="Category_Code",
    how="left",
)

# Reorder columns to include Category_Label_Short
columns_order = [
    "Category_Code",
    "Category_Label_Short",
    "category1",
    "category2",
    "category3",
    "category4",
    "Data_Type",
    "Model_Type",
    "AIC_logit",
    "AIC",
    "AIC_diff",
    "HHI_diff",
]
table_category_level = table_category_level[columns_order]

# Create table_category_tex by keeping only specified columns
table_category_tex = table_category_level[
    ["Category_Label_Short", "Data_Type", "Model_Type", "AIC_logit", "AIC", "AIC_diff"]
]

# Data
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
kde_path = f"{output_directory}/aic_histogram.png"
plt.savefig(kde_path, dpi=300)

# Display the plot
plt.show()

# selected_columns = ['Category_Label_Short', 'Data_Type', 'Model_Type', 'AIC_logit', 'AIC', 'AIC_diff']
selected_columns = [
    "Category_Label_Short",
    "Data_Type",
    "Model_Type",
    "AIC_diff",
    "HHI_diff",
]
df_selected = table_category_level[selected_columns]

# Add a number prefix to the 'Category_Label_Short' values
df_selected["Category_Label_Short"] = [
    f"{i+1}. {category}"
    for i, category in enumerate(df_selected["Category_Label_Short"])
]

# Rename columns
df_selected.columns = [
    "Category",
    "Data Type",
    "Model Type",
    "$\\Delta$ AIC",
    "$\\Delta$ HHI",
]
df_selected.loc[:, "Category"] = df_selected["Category"].str.replace("&", r"\&")

# Format Delta HHI column as a percentage
df_selected["$\\Delta$ HHI"] = df_selected["$\\Delta$ HHI"].apply(
    lambda x: f"{x:.1f}\\%"
)

# Specify the column format to set a custom width for the first column
column_format = "p{6cm}" + "c" * (
    len(df_selected.columns) - 1
)  # Set the width of the first column to 5cm

# Index rows from 1 to C for latex table
df_selected.index = range(1, len(df_selected) + 1)

# Create a LaTeX table from the df_selected DataFrame
latex_table = df_selected.to_latex(
    buf=None,  # Do not save directly, return the LaTeX string
    float_format="{:.1f}".format,  # Format numbers to 3 decimal places
    index=False,  # Include row indices
    escape=False,  # Allow special characters in the table (like "_")
    column_format=column_format,  # Apply the custom column format
)

latex_table = latex_table.strip()

# Save the LaTeX table to a .tex file
with open(f"{output_directory}/category_level_results.tex", "w") as f:
    f.write(latex_table)

# SCRIPT 3

# Define file paths
wide_table_path = f"{output_directory}/wide_table.csv"
output_file = f"{output_directory}/akaike_weights_all_categories.png"

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
# filtered_table.loc[:, "Category_Label_Short"] = filtered_table["Category_Label_Short"].astype(str)
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
mpl.rcParams["hatch.linewidth"] = 0.1  # Adjust this value for thinner or thicker lines

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(
    axis="y", labelsize=8
)  # Adjust the label size (10 is an example, reduce further if needed)
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
output_file = f"{output_directory}/akaike_weights_all_categories.png"
plt.savefig(output_file, dpi=300)

# Display the plot
plt.show()
