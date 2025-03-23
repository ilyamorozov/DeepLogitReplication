import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir = "data/experiment/intermediate/survey_responses"
data = pd.read_csv(f"{data_dir}/ebook_survey_data_cleaned.csv")

output_dir = "data/experiment/output/appendix_figures"
os.makedirs(output_dir, exist_ok=True)

# A1

# Convert time in seconds to minutes for duration and for choice task timings
data["duration_minutes"] = data["duration_(in_seconds)"] / 60
data["choice_time_avg_minutes"] = (
    (data["timing_choice1_page_submit"] + data["timing_choice2_page_submit"]) / 2
) / 60

# Truncate values greater than 30 minutes
data["duration_minutes_truncated"] = data["duration_minutes"].apply(
    lambda x: 30 if x > 30 else x
)
data["choice_time_avg_minutes_truncated"] = data["choice_time_avg_minutes"].apply(
    lambda x: 30 if x > 30 else x
)

avg_total_time = data["duration_minutes"].mean()
avg_choice_time = data["choice_time_avg_minutes"].mean()

plt.figure(figsize=(8, 5))
plt.hist(
    data["duration_minutes_truncated"],
    bins=20,
    color="#1f77b4",
    edgecolor="black",
    alpha=0.7,
    weights=[1 / len(data)] * len(data),
)  # Normalize Y-axis to fraction
plt.title(
    f"Total Time Spent on Study (Average = {avg_total_time:.2f} min)",
    fontsize=15,
    pad=20,
)
plt.xlabel("Study Time (Minutes)", fontsize=14)
plt.ylabel("Fraction", fontsize=14)
plt.xticks([0, 5, 10, 15, 20, 25, 30], [str(x) for x in range(0, 30, 5)] + ["30+"])
plt.tight_layout()


plt.savefig(f"{output_dir}/total_time_spent.png")


# Plotting the histogram for time spent on the two choice tasks
plt.figure(figsize=(8, 5))
plt.hist(
    data["choice_time_avg_minutes_truncated"],
    bins=20,
    color="#4c72b0",
    edgecolor="black",
    alpha=0.7,
    weights=[1 / len(data)] * len(data),
)  # Normalize Y-axis to fraction
plt.title(
    f"Time Spent on Choice Tasks (Average = {avg_choice_time:.2f} min)",
    fontsize=14,
    pad=20,
)
plt.xlabel("Time on Choice Tasks (Minutes)", fontsize=12)
plt.ylabel("Fraction", fontsize=12)
plt.xticks([0, 5, 10, 15, 20, 25, 30], [str(x) for x in range(0, 30, 5)] + ["30+"])
plt.tight_layout()

plt.savefig(f"{output_dir}/time_spent_choice_tasks.png")

# A2

# Mapping ranks to genres based on the user-provided genre rankings
genre_ranking_map = {
    1: "Mystery",
    2: "Romance",
    3: "Sci-fi",
    4: "Biographies",
    5: "Self-help",
}

# Create a dictionary for mapping selected book IDs to genres
id_genre_mapping = {
    3: "Mystery",
    4: "Mystery",
    8: "Mystery",
    9: "Mystery",
    21: "Sci-fi/Fantasy",
    23: "Sci-fi/Fantasy",
    29: "Sci-fi/Fantasy",
    45: "Self-help",
    46: "Self-help",
    47: "Self-help",
}


# Create a function to get the self-reported top ranked genre
def get_top_ranked_genre(row):
    for genre_rank, genre_name in genre_ranking_map.items():
        if row.get(f"rank_genres_{genre_rank}", np.nan) == 1:
            return genre_name
    return None


# Function to calculate the Wald confidence interval for a proportion
def wald_ci(proportion, n, z=1.96):
    # Proportion is the fraction for the genre, n is the sample size
    se = np.sqrt(proportion * (1 - proportion) / n)
    lower_bound = proportion - z * se
    upper_bound = proportion + z * se
    return lower_bound, upper_bound


data["self_reported_top_genre"] = data.apply(get_top_ranked_genre, axis=1)

# Define colors from light to dark blue for the bars
color_map = ["#d6e6f2", "#6baed6", "#2171b5"]

# Map the chosen product_id to the corresponding genre
data["first_choice_genre"] = data["selected_id_1"].map(id_genre_mapping)

# Create a cross-tabulation of the genres
crosstab_data = pd.crosstab(data["self_reported_top_genre"], data["first_choice_genre"])

# Get the total number of observations for each self-reported top genre (row sums)
total_per_self_reported = crosstab_data.sum(axis=1)

# Normalize the cross-tabulation to get the proportions
proportion_data = crosstab_data.div(total_per_self_reported, axis=0)


# Create a new DataFrame to store the confidence intervals
ci_data = pd.DataFrame(index=proportion_data.index, columns=proportion_data.columns)


# Calculate the confidence intervals for each genre combination
for genre in proportion_data.columns:
    for self_reported_genre in proportion_data.index:
        proportion = proportion_data.loc[self_reported_genre, genre]
        n = total_per_self_reported[self_reported_genre]
        lower, upper = wald_ci(proportion, n)
        ci_data.loc[self_reported_genre, genre] = (lower, upper)


# Function to extract symmetric error bars (assuming symmetric confidence intervals)
def get_symmetric_error_bars(row, genre):
    lower, upper = row[genre]
    error = upper - row[genre][0]  # Assuming symmetric confidence intervals
    return error  # Return a single error value since it's symmetric


# Define the positions of the bars for the grouped chart
n_genres = len(proportion_data.index)
bar_width = 0.15  # Reduced bar width for better gap
x = np.arange(n_genres)

# Adjust the gap between bars by further reducing the width and increasing the offset
gap = 0.03  # Extra gap between bars


# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each genre with symmetric confidence intervals
for genre_index, genre in enumerate(["Mystery", "Sci-fi/Fantasy", "Self-help"]):
    # Get the proportions and symmetric error bars
    proportions = proportion_data[genre].values
    errors = [
        get_symmetric_error_bars(ci_data.loc[row], genre)
        for row in proportion_data.index
    ]

    # Plot the bars with symmetric error bars
    ax.bar(
        x + (genre_index - 1) * (bar_width + gap),
        proportions,
        width=bar_width,
        edgecolor="black",
        label=genre,
        color=color_map[genre_index],
        yerr=errors,
        capsize=3,
    )

# Add title and labels
ax.set_title("Genre of First Choice vs Self-Reported Top Ranked Genre", fontsize=18)
ax.set_xlabel("Most preferred genre (self-reported)", fontsize=15)
ax.set_ylabel("Probability", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(proportion_data.index)
ax.set_ylim(-0.05, 1.05)  # Adjust as needed; this sets the lower limit to -0.05


ax.tick_params(axis="x", labelsize=13, width=1.5)
ax.tick_params(axis="y", labelsize=13, width=1.5)

# Remove upper and left spines (borders)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.5)  # Set thickness for the bottom spine
ax.spines["left"].set_linewidth(1.5)  # Set thickness for the right spine

# Modify the legend
# plt.legend(title='Genre of Choice', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, frameon=False)
plt.legend(
    bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3, frameon=False, fontsize=14
)


# Adjust the legend
# handles, labels = ax.get_legend_handles_labels()  # Get the current legend handles and labels
# legend = ax.legend(handles, labels, bbox_to_anchor=(0.5, -0.15), loc='upper center',
#                   ncol=3, frameon=False, handletextpad=1.1)

# Add the legend title below the markers (manually placing it)
# ax.text(0.5, -0.3, 'Genre of Choice', ha='center', va='center', transform=ax.transAxes, fontsize=10)


plt.tight_layout()
plt.savefig(f"{output_dir}/first_choice_genre_vs_self_reported_genre.png", dpi=300)

# A3

data["second_choice_genre"] = data["selected_id_2"].map(id_genre_mapping)

# Cross-tabulation of first-choice vs second-choice genres
crosstab_data = pd.crosstab(data["first_choice_genre"], data["second_choice_genre"])

# Get the total number of observations for each first-choice genre (row sums)
total_first_choice = crosstab_data.sum(axis=1)

# Normalize the cross-tabulation to get the proportions
proportion_data = crosstab_data.div(total_first_choice, axis=0)

# Calculate the confidence intervals for each second-choice genre within each first-choice genre
ci_data = pd.DataFrame(index=proportion_data.index, columns=proportion_data.columns)

for first_choice_genre in proportion_data.index:
    n = total_first_choice[
        first_choice_genre
    ]  # Total number of observations for the first-choice genre
    for second_choice_genre in proportion_data.columns:
        proportion = proportion_data.loc[first_choice_genre, second_choice_genre]
        lower, upper = wald_ci(proportion, n)
        ci_data.loc[first_choice_genre, second_choice_genre] = (lower, upper)


# Function to extract the symmetric error bars (assuming symmetric confidence intervals)
def get_symmetric_error_bars(row, genre):
    lower, upper = row[genre]
    error = upper - row[genre][0]  # Assuming symmetric confidence intervals
    return error  # Return a single error value since it's symmetric


# Define the positions of the bars for the grouped chart
n_genres = len(proportion_data.index)
bar_width = 0.15  # Reduced bar width for better gap
x = np.arange(n_genres)

# Adjust the gap between bars by further reducing the width and increasing the offset
gap = 0.025  # Extra gap between bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each genre with adjusted color gradient and confidence intervals
for genre_index, genre in enumerate(["Mystery", "Sci-fi/Fantasy", "Self-help"]):
    # Get the proportions and symmetric error bars
    proportions = proportion_data[genre].values
    errors = [
        get_symmetric_error_bars(ci_data.loc[row], genre)
        for row in proportion_data.index
    ]

    # Plot the bars with symmetric error bars
    ax.bar(
        x + (genre_index - 1) * (bar_width + gap),
        proportions,
        width=bar_width,
        label=genre,
        edgecolor="black",
        color=color_map[genre_index],
        yerr=errors,
        capsize=3,
    )

# Add title and labels
ax.set_title("Genre of First Choice vs. Genre of Second Choice", fontsize=18)
ax.set_xlabel("Genre of First Choice", fontsize=15)
ax.set_ylabel("Probability", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(proportion_data.index)
ax.set_ylim(-0.05, 1.05)  # Adjust as needed; this sets the lower limit to -0.05

ax.tick_params(axis="x", labelsize=13, width=1.5)
ax.tick_params(axis="y", labelsize=13, width=1.5)

# Remove upper and left spines (borders)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.5)  # Set thickness for the bottom spine
ax.spines["left"].set_linewidth(1.5)  # Set thickness for the right spine

# Modify the legend to be below the plot and remove the box
# plt.legend(title='Genre of Second Choice', bbox_to_anchor=(0.5, -0.15), loc='upper center',
#           ncol=3, frameon=False)
# Modify the legend to be below the plot and remove the box
plt.legend(
    bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3, frameon=False, fontsize=14
)


plt.tight_layout()
plt.savefig(f"{output_dir}/first_choice_genre_vs_second_choice_genre.png", dpi=300)
