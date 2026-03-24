#  Replication Package for the paper "Demand Estimation with Text and Image Data"


## Overview

The code in this replication package reproduces the tables, figures, and in-text results in the paper. The analysis draws on two data sources: (1) a choice experiment with 10 ebooks and approximately 9,265 participants recruited via Prolific, and (2) purchase data from 40 product categories on Amazon.com using the Comscore Web Behavior Panel (2019-2020). Using pre-trained deep learning models, the code extracts embeddings from product images and textual descriptions, applies Principal Component Analysis (PCA) to reduce dimensionality, and estimates mixed logit demand models that incorporate these embeddings as product characteristics.

The replication package uses three software environments: Python, R, and Stata. The experiment pipeline is fully reproducible from the data provided in this package. The Comscore pipeline requires proprietary data that cannot be included.

## Data Availability and Provenance Statements

The authors certify that they have legitimate access to and permission to use all data used in this manuscript. The experiment data collected by the authors are provided as part of this replication package and are redistributable under the GPLv3 license (see `LICENSE`). The Comscore Web Behavior Panel data are proprietary and cannot be redistributed.

### Summary of Data Availability

| Data | Files | Location | Provided | Source |
|------|-------|----------|----------|--------|
| Ebook metadata | `books.csv` | `data/experiment/books/` | Yes | Collected by authors from Amazon |
| Book cover images | 10 JPG files | `data/experiment/books/book_covers/` | Yes | Collected by authors from Amazon |
| Survey responses | `ebook_survey_data_edited.csv` (18 MB) | `data/experiment/survey_responses/` | Yes | Collected by authors via Prolific |
| Comscore purchases | `amazon_crosswalk.dta`, category/price .dta files | `data/comscore/input/purchases/`, `categories/`, `prices/`, `selected_categories/` | No | Comscore (2019-2020); Keepa.com (prices) |
| Category list | `comscore_categories.csv` | `data/comscore/input/` | No | Greminger, Huang, and Morozov (2023) |
| Selected product ASINs | `asin_list.csv` per category | `data/comscore/input/selected_products/[category]/` | No | Constructed by authors |
| Product images and text | Images, descriptions, reviews per category | `data/comscore/input/text_and_images/[category]/` | No | Collected by authors from Amazon |

### Details on Experiment Data (provided)

The experiment data were collected by the authors. Each participant completed two choice tasks, selecting ebooks from a set of 10 alternatives with randomized prices and rankings.

- **`data/experiment/books/books.csv`**: CSV file containing metadata for the 10 books used in the experiment. Columns include ASIN, title, author, description, five user reviews, genre, publication year, and page count. 10 rows.
- **`data/experiment/books/book_covers/`**: 10 JPEG images of ebook covers, named by Amazon ASIN (e.g., `B098MHBF23.jpg`).
- **`data/experiment/survey_responses/ebook_survey_data_edited.csv`**: Survey responses from approximately 9,265 Prolific participants. Contains participant identifiers, demographics, first and second choice selections, randomized prices, and book rankings. Approximately 18 MB.

### Details on Comscore Data (not provided)

The paper uses purchase data from the 2019-2020 Comscore Web Behavior Panel combined with product images and text scraped from Amazon product detail pages, and daily price histories from Keepa.com. The category classification follows the dataset constructed by Greminger, Huang, and Morozov (2023), who classify over 12 million unique products from Amazon.com into narrowly defined categories.

**Access.** Researchers interested in accessing the Comscore data must negotiate a data use agreement with Comscore, Inc. (https://www.comscore.com/). The data are proprietary and subject to licensing fees. Product images and text were scraped by the authors from Amazon product detail pages. Price histories were obtained from Keepa.com (https://keepa.com/), a commercial service.

The Comscore pipeline uses data from two stages. First, the Stata script (`1_comscore_build_dataset.do`) reads raw proprietary Comscore files and produces intermediate files. Second, the Python scripts (2-6) read from the input and intermediate directories.

#### Raw Comscore files (read by the Stata script)

The Stata script expects the following raw files, which must be obtained from Comscore and placed in appropriate directories:

| File | Description |
|------|-------------|
| `selected_categories/master_list_categories.dta` | Master list of product category codes (Stata format). Each row is a category with a `cat_code4` variable identifying the 8-digit category code. |
| `purchases/amazon_crosswalk.dta` | Purchase transaction records linking Comscore panelists to Amazon ASINs. Contains `product_asin`, `machine_id` (panelist identifier), and `event_time` (purchase timestamp). |
| `categories/all_variations2.dta` | ASIN variation/pooling mappings. Maps product variation ASINs to canonical `asin_pooled` identifiers so that different listings of the same product are grouped together. |
| `prices/prices.dta`, `prices_extra.dta`, `prices_avg.dta` | Daily price data per product, obtained from Keepa.com (Stata format). Used to construct price matrices for each purchase occasion. |

#### Input files (read by Python scripts 2-6)

The Python scripts expect the following files under `data/comscore/input/`:

| File | Description |
|------|-------------|
| `comscore_categories.csv` | Single-column CSV listing the 40 category codes used in the analysis. Read by scripts 2, 3, 4, 5, and 6. |
| `selected_products/[category_code]/asin_list.csv` | One subdirectory per category. Each `asin_list.csv` contains an `asin` column listing Amazon ASINs for the products included in estimation. Read by the Stata script (which uses only the `asin` column to build choice data) and Python scripts 2 and 3. |
| `text_and_images/[category_code]/product_descriptions.csv` | Product descriptions for each category. Columns: `asin`, `product description`, up to 10 bullet-point columns (`0`-`9`), and `product_title`. Used by script 2 for text embedding generation and by script 6 for product title lookups. |
| `text_and_images/[category_code]/reviews.csv` | Customer reviews for each category. Columns: `asin`, `product_title`, `rating`, `review_title`, `review_text`, `review-links`. Up to 100 most recent reviews per product. Used by script 2 for text embedding generation. |
| `text_and_images/[category_code]/images/` | Product images (PNG or JPG) named by ASIN (e.g., `B07K1RZWMC.png`). Default product photos from Amazon product detail pages. Used by script 2 for image embedding generation. |

#### Intermediate files (generated by scripts 1-3, read by scripts 4-6)

The intermediate directory (`data/comscore/intermediate/`) contains one subdirectory per category, automatically generated by the pipeline. Script 1 (Stata) produces choice data in long format (`long_format_data.csv`), purchase and price matrices in wide format (`matrix_purchases.csv`, `matrix_prices.csv`), and an ASIN-to-product-ID crosswalk (`asin_to_item_crosswalk.csv`). Script 2 generates image and text embeddings (stored as `.npy` and `.csv` files under `embeddings/images/` and `embeddings/texts/`). Script 3 applies PCA to produce principal component CSVs (under `principal_components/`). These intermediate files are consumed by scripts 4-6 for estimation, elasticity computation, and visualization.

## Computational Requirements

### Software Requirements

- **Python**
  - Packages listed in `requirements.txt`. Install via: `pip install -r requirements.txt`
  - Key packages: `tensorflow==2.18.0`, `keras==3.6.0`, `torch==2.5.1`, `sentence-transformers==3.3.0`, `xlogit==0.2.7`, `pylogit==1.0.1`, `scikit-learn==1.5.2`, `pandas==2.2.3`, `matplotlib==3.9.4`, `numpy==2.0.2`, `scipy==1.13.1`
- **R** (any recent version, e.g., 4.x)
  - Packages: `data.table`, `tidyverse`, `readxl` (installed automatically by the R script)
  - Required only for `0_experiment_build_dataset.R`
- **Stata**
  - Required only for `1_comscore_build_dataset.do` (Comscore pipeline)
  - Not needed if only replicating the experiment results
- **Internet access** required on first run: pre-trained model weights for VGG16, VGG19, ResNet50, Xception, and InceptionV3 are downloaded automatically via Keras/TensorFlow. The Universal Sentence Encoder is downloaded via TensorFlow Hub. The Sentence Transformer model is downloaded via HuggingFace Hub.

### Controlled Randomness

Random seeds are set in the estimation and embedding generation scripts. Due to differences in floating-point arithmetic across hardware and software versions (particularly in deep learning frameworks and numerical optimization), minor numerical differences may occur in embedding values and estimated coefficients. The qualitative results and conclusions should be robust to such differences.

### Memory, Storage, and Computation

- **Storage**: The replication package as provided is approximately 250 MB. With all intermediate and output files generated, expect approximately 500 MB for the experiment pipeline. The Comscore pipeline may require several additional GB.
- **Memory**: At least 8 GB RAM recommended. A GPU is not required but will accelerate embedding generation.
- **Computation**: Code was run on a high-performance computing cluster, with multiple cores used for some estimation scripts.

## Description of Programs/Code

### Directory Structure

```
DeepLogitReplication/
├── README.md
├── LICENSE                             # GPLv3
├── requirements.txt                    # Python dependencies
├── src/
│   ├── config.ini                      # Centralized path configuration
│   ├── helper_functions/               # Shared utility modules
│   │   ├── embeddings/                 # Embedding generation, PCA, loading
│   │   ├── estimation/                 # Mixed logit estimation, specification generation
│   │   ├── file_structure/             # Config path resolution
│   │   └── visualization/              # Figure and table generation utilities
│   ├── replicate_experiment/           # Experiment pipeline (scripts 0-10)
│   └── replicate_comscore/             # Comscore pipeline (scripts 1-6)
├── data/
│   ├── experiment/                     # Provided experiment data
│   └── comscore/                       # Proprietary Comscore data (not provided)
├── temp/                               # Experiment intermediate files (embeddings, principal components)
└── output/                             # Final tables, figures, and estimation results
```

### Experiment Pipeline (11 scripts, numbered 0-10)

| # | Script | Language | Description |
|---|--------|----------|-------------|
| 0 | `0_experiment_build_dataset.R` | R | Cleans and processes raw survey data into choice data for logit estimation |
| 1 | `1_experiment_generate_embeddings.py` | Python | Extracts image embeddings (VGG16, VGG19, ResNet50, Xception, InceptionV3) and text embeddings (TF-IDF, Count, USE, Sentence Transformer) from book metadata and covers |
| 2 | `2_experiment_prepare_embeddings.py` | Python | Applies PCA to reduce embedding dimensionality to principal components |
| 3 | `3_experiment_estimation_mixed_logit.py` | Python | Estimates mixed logit models across all embedding specifications using Algorithm 1 from the paper |
| 4 | `4_experiment_visualizations.py` | Python | Generates model comparison scatter plots and principal component visualizations |
| 5 | `5_experiment_visualize_transitions.py` | Python | Computes diversion matrices and generates implied substitution tables |
| 6 | `6_experiment_table_model_results.py` | Python | Produces LaTeX tables summarizing model selection results |
| 7 | `7_experiment_merger_simulations.py` | Python | Runs counterfactual merger simulations computing equilibrium prices under joint ownership |
| 8 | `8_experiment_visualize_mergers.py` | Python | Visualizes merger simulation results |
| 9 | `9_experiment_appendix_figures.py` | Python | Generates appendix figures: time distributions, genre patterns, variance explained |
| 10 | `10_experiment_compute_correlations.py` | Python | Computes AIC-RMSE correlations reported in the text |

### Comscore Pipeline (7 scripts, numbered 1-6)

| # | Script | Language | Description |
|---|--------|----------|-------------|
| 1 | `1_comscore_build_dataset.do` | Stata | Builds choice datasets from Comscore purchases, prices, and category data for each of the 40 categories |
| 2 | `2_comscore_generate_embeddings.py` | Python | Generates image and text embeddings for Amazon products across all categories |
| 3 | `3_comscore_prepare_embeddings.py` | Python | Applies PCA to reduce Comscore embedding dimensionality |
| 4 | `4_comscore_estimation_mixed_logit.py` | Python | Estimates mixed logit models for each of the 40 product categories |
| 5 | `5_comscore_elasticities.py` | Python | Computes own- and cross-price elasticities and diversion ratios via Monte Carlo simulation |
| 6 | `6_comscore_visualize.py` | Python | Produces all Comscore tables and figures |

All file paths are centralized in `src/config.ini`. Each section (e.g., `[EXPERIMENT_1]`, `[COMSCORE_2]`) defines input, intermediate, and output paths for the corresponding script. No path editing is necessary if scripts are run from the project root directory.

## Instructions to Replicators

### Replicating the Experiment Results (fully reproducible)

1. Ensure Python, R (with `data.table`, `tidyverse`, `readxl`), and `pip` are installed.
2. From the project root directory, install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the following scripts in order from the project root directory:
   ```bash
   Rscript src/replicate_experiment/0_experiment_build_dataset.R
   python -m src.replicate_experiment.1_experiment_generate_embeddings
   python -m src.replicate_experiment.2_experiment_prepare_embeddings
   python -m src.replicate_experiment.3_experiment_estimation_mixed_logit
   python -m src.replicate_experiment.4_experiment_visualizations
   python -m src.replicate_experiment.5_experiment_visualize_transitions
   python -m src.replicate_experiment.6_experiment_table_model_results
   python -m src.replicate_experiment.7_experiment_merger_simulations
   python -m src.replicate_experiment.8_experiment_visualize_mergers
   python -m src.replicate_experiment.9_experiment_appendix_figures
   python -m src.replicate_experiment.10_experiment_compute_correlations
   ```
4. Output will be written to `output/experiment/` (figures in `figures/`, tables in `tables/`).

**Note:** Script 1 downloads pre-trained deep learning model weights on first run (requires internet access). Subsequent runs use cached models.

### Replicating the Comscore Results (requires proprietary data)

1. Obtain the Comscore Web Behavior Panel data and Amazon product data (see Data Availability section above).
2. Place data files in `data/comscore/input/` following the expected directory structure described above.
3. Ensure Stata is installed.
4. From the directory `src/replicate_comscore/`, run the Stata script:
   ```bash
   stata-mp -b do 1_comscore_build_dataset.do
   ```
5. From the project root directory, run the Python scripts in order:
   ```bash
   python -m src.replicate_comscore.2_comscore_generate_embeddings
   python -m src.replicate_comscore.3_comscore_prepare_embeddings
   python -m src.replicate_comscore.4_comscore_estimation_mixed_logit
   python -m src.replicate_comscore.5_comscore_elasticities
   python -m src.replicate_comscore.6_comscore_visualize
   ```
6. Output will be written to `output/comscore/` (figures in `figures/`, tables in `tables/`).

## List of Tables and Programs

The provided code reproduces all tables and figures in the paper. Tables and figures from the Comscore analysis require proprietary data not included in this package.

| Figure/Table | Program | Output file | Note |
|---|---|---|---|
| Table 1 (p. 11) | `src/replicate_experiment/6_experiment_table_model_results.py` | `output/experiment/tables/selected_models_aic_summary.tex` | |
| Table 2 (p. 16) | `src/replicate_experiment/5_experiment_visualize_transitions.py` | `output/experiment/tables/first_three_products.tex` | |
| Table 3 (p. 20) | `src/replicate_comscore/6_comscore_visualize.py` | `output/comscore/tables/main_electronic_categories_AIC.tex` | Requires Comscore data |
| Figure 2 (p. 12) | `src/replicate_experiment/4_experiment_visualizations.py` | `output/experiment/figures/selected_model_scatters/second_choice_rmse_best_rmse.png` | |
| Figure 4 (p. 15) | `src/replicate_experiment/4_experiment_visualizations.py` | `output/experiment/figures/selected_model_scatters/PC1_PC2.png` | |
| Figure 6 (p. 21) | `src/replicate_comscore/6_comscore_visualize.py` | `output/comscore/figures/diversion_plain_vs_our.png` | Requires Comscore data |
| Figure 7 (p. 22) | `src/replicate_comscore/6_comscore_visualize.py` | `output/comscore/figures/akaike_weights_all_categories.png` | Requires Comscore data |
| Figure A1 (p. 34) | `src/replicate_experiment/8_experiment_visualize_mergers.py` | `output/experiment/figures/merger_graphs/merger_simulation_8.png` | |
| Figure A2 (p. 38) | `src/replicate_experiment/9_experiment_appendix_figures.py` | `output/experiment/figures/appendix_figures/total_time_spent.png`, `time_spent_choice_tasks.png` | Top and bottom panels |
| Figure A3 (p. 39) | `src/replicate_experiment/9_experiment_appendix_figures.py` | `output/experiment/figures/appendix_figures/first_choice_genre_vs_self_reported_genre.png` | |
| Figure A4 (p. 39) | `src/replicate_experiment/9_experiment_appendix_figures.py` | `output/experiment/figures/appendix_figures/first_choice_genre_vs_second_choice_genre.png` | |
| Figure A5 (p. 40) | `src/replicate_experiment/9_experiment_appendix_figures.py` | `output/experiment/figures/appendix_figures/explained_variance_cumulative_by_type.png` | |
| Figure A6 (p. 40) | `src/replicate_comscore/6_comscore_visualize.py` | `output/comscore/figures/aic_histogram.png` | Requires Comscore data |
| Table A1 (p. 41) | `src/replicate_experiment/6_experiment_table_model_results.py` | `output/experiment/tables/model_validation_results.tex` | |
| Table A2 (p. 42) | `src/replicate_comscore/6_comscore_visualize.py` | `output/comscore/tables/diversion_matrices/plain_logit_diversion_with_titles_13060404.tex` | Requires Comscore data |
| Table A3 (p. 42) | `src/replicate_comscore/6_comscore_visualize.py` | `output/comscore/tables/diversion_matrices/attr_based_pca_diversion_with_titles_13060404.tex` | Requires Comscore data |
| Table A4 (p. 42) | `src/replicate_comscore/6_comscore_visualize.py` | `output/comscore/tables/diversion_matrices/pca_diversion_with_titles_13060404.tex` | Requires Comscore data |
| Table A5 (p. 43) | `src/replicate_comscore/6_comscore_visualize.py` | `output/comscore/tables/category_level_results_2025-08-21.tex` | Requires Comscore data |

## References

Compiani, G., I. Morozov, and S. Seiler (2023): "Demand Estimation with Text and Image Data," Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4588941


Greminger, R., Y. Huang, and I. Morozov (2023): "Make Every Second Count: Time Allocation in Online Shopping," Working Paper.

Comscore, Inc. (2019-2020): "Web Behavior Panel." https://www.comscore.com/

Keepa.com: "Amazon Price Tracker." https://keepa.com/
