# Replication Package for the paper "Demand Estimation with Text and Image Data"

Please use a version of Python 3.9.x to run the scripts in this replication package. Use the `requirements.txt` file to install the required packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

After configuring the environment, you can reproduce the results by running the following scripts:

## Reproducing the experimental results

In the project root directory, run the following:

```bash
Rscript src/replicate_experiment/0_experiment_build_dataset.R
python -m src.replicate_experiment.1_experiment_generate_embeddings
python -m src.replicate_experiment.2_experiment_prepare_embeddings
python -m src.replicate_experiment.3_experiment_estimation_mixed_logit
python -m src.replicate_experiment.4_experiment_visualizations
python -m src.replicate_experiment.5_experiment_visualize_transitions
python -m src.replicate_experiment.6_experiment_table_model_results
python -m src.replicate_experiment.7_experiment_compute_consumer_welfare
python -m src.replicate_experiment.8_experiment_merger_simulations
python -m src.replicate_experiment.9_experiment_visualize_mergers
python -m src.replicate_experiment.10_experiment_appendix_figures
```

Script 1 takes the unstructured book data in `data/experiment/input/books` and generate embeddings for the text and image and store them in `data/experiment/intermediate/embeddings`. Script 2 runs PCA on the embeddings to prepare them for the estimation procedure. Script 3 estimates the mixed logit model and stores the estimation results in `data/experiment/output/estimation_results` as an XLSX file. Script 4 generates visualizations of the estimation results and stores them in `data/experiment/output/figures`.

## Reproducing the comscore results

In the directory `src/replicate_comscore`, run the following Stata do file:

```bash
stata-mp -b do 1_comscore_build_dataset.do
```

In the project root directory, run the following:

```bash
python -m src.replicate_comscore.2_comscore_generate_embeddings
python -m src.replicate_comscore.3_comscore_prepare_embeddings
python -m src.replicate_comscore.4_comscore_estimation_mixed_logit
python -m src.replicate_comscore.5_comscore_visualize
python -m src.replicate_comscore.6_comscore_elasticities
```
