# Bringing BERT to the field: Transformer models for gene expression prediction in maize
Predicting gene expression levels from upstream promoter regions using deep learning. Collaboration between IACS and Inari.

---

## Directory Setup

**`scripts/`: directory for production code**

- [`0-data-loading-processing/`](https://github.com/shanmukh0504/FloraBERT/tree/master/scripts/0-data-loading-processing):
  - [`01-gene-expression.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/0-data-loading-processing/01-gene-expression.py): downloads and processes gene expression data and saves into "B73_genex.txt".
  - [`02-download-process-db-data.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/0-data-loading-processing/02-download-process-db-data.py): downloads and processes gene sequences from a specified database: 'Ensembl', 'Maize', 'Maize_addition', 'Refseq'
  - [`03-combine-databases.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/0-data-loading-processing/03-combine-databases.py): combines all the downloaded sequences within all the databases
  - [`04a-merge-genex-maize_seq.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/0-data-loading-processing/04a-merge-genex-maize_seq.py):
  - [`04b-merge-genex-b73.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/0-data-loading-processing/04b-merge-genex-b73.py):
  - [`05a-cluster-maize_seq.sh`](scripts/0-data-loading-processing/05a-cluster-maize_seq.sh): clusters the promoter sequences into groups with up to 80% sequence identity, which may be interpreted as paralogs
  - [`05b-train-test-split.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/0-data-loading-processing/05-train-test-split.py): divides the promoter sequences into train and test sets, avoiding a set of pairs that indicate close relations ("paralogs")
  - [`06_transformer_preparation.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/0-data-loading-processing/06_transformer_preparation.py):
  - [`07_train_tokenizer.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/0-data-loading-processing/07_train_tokenizer.py): training byte-level BPE for RoBERTa model
- [`1-modeling/`](https://github.com/shanmukh0504/FloraBERT/tree/master/scripts/1-modeling)
  - [`pretrain.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/1-modeling/pretrain.py): training the FLORABERT base using a masked language modeling task. Type `python scripts/1-modeling/pretrain.py --help` to see command line options, including choice of dataset and whether to warmstart from a partially trained model. Note: not all options will be used by this script.
  - [`finetune.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/1-modeling/finetune.py): training the FLORABERT regression model (including newly initialized regression head) on multitask regression for gene expression in all 10 tissues. Type `python scripts/1-modeling/finetune.py --help` to see command line options; mainly for specifying data inputs and output directory for saving model weights.
  - [`evaluate.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/1-modeling/evaluate.py): computing metrics for the trained FLORABERT model
- [`2-feature-visualization/`](https://github.com/shanmukh0504/FloraBERT/tree/master/scripts/2-feature-visualization)`
  - [`embedding_vis.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/scripts/2-feature-visualization/embedding_vis.py): computing a sample of BERT embeddings for the testing data and saving to a tensorboard log. Can specify how many embeddings to sample with `--num-embeddings XX` where `XX` is the number of embeddings (must be integer).

## Current build checkpoints

The repository currently includes the processed NAM expression dataset, tokenizer files, model code, and smoke-test scripts. Full trained transformer checkpoint weights are not checked in under `models/transformer`, so evaluation of a trained model requires either training locally or adding downloaded checkpoint files.

### Project status

FloraBERT is currently set up as a reproducible research pipeline:

- Processed NAM promoter/expression data is available locally.
- The byte-level BPE tokenizer is available locally.
- Model construction, data loading, training, checkpoint loading, prediction, evaluation, and report generation are verified.
- Local smoke checkpoints can be trained and evaluated end-to-end.

The smoke checkpoints are intentionally tiny validation models. They prove that the pipeline works, but their metrics should not be presented as final biological performance. A final result requires a longer run with a larger training budget and, ideally, warm-starting from a pretrained language-model checkpoint.

Set up a local environment:

```bash
make python_requirements
```

Run checkpoint checks:

```bash
.venv/bin/python scripts/check_project_health.py
.venv/bin/python scripts/smoke_test_model.py
.venv/bin/python scripts/smoke_train.py
.venv/bin/python scripts/smoke_evaluate.py
```

Run the one-command local demo:

```bash
.venv/bin/python scripts/run_demo_pipeline.py
```

Regenerate the smoke checkpoint during the demo:

```bash
.venv/bin/python scripts/run_demo_pipeline.py --train-smoke
```

Run a prediction demo:

```bash
.venv/bin/python scripts/predict.py --gene-id Zm00001eb002390
```

Use a trained checkpoint when available:

```bash
.venv/bin/python scripts/predict.py \
  --gene-id Zm00001eb002390 \
  --checkpoint models/transformer/prediction-model
```

Train and evaluate a small local checkpoint:

```bash
.venv/bin/python scripts/1-modeling/train_prediction_smoke.py \
  --max-steps 5 \
  --nshards 20 \
  --batch-size 2

.venv/bin/python scripts/predict.py \
  --gene-id Zm00001eb002390 \
  --checkpoint models/transformer/prediction-model-smoke

.venv/bin/python scripts/1-modeling/evaluate_checkpoint.py \
  --checkpoint models/transformer/prediction-model-smoke \
  --nshards 20 \
  --output output/model_eval/prediction-model-smoke.csv \
  --report-dir output/reports/prediction-model-smoke
```

Local checkpoints under `models/transformer/` and generated outputs under `output/` are ignored by git.

Run the main fine-tuning path on a small shard:

```bash
.venv/bin/python scripts/1-modeling/finetune.py \
  --output-dir models/transformer/prediction-model-finetune-smoke \
  --pretrained-model '' \
  --nshards 50 \
  --max-steps 3 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --n-workers 1 \
  --learning-rate 1e-4
```

For a longer local run, reduce or remove `--nshards` and increase `--max-steps`.

The checkpoint evaluation command writes:

- `metrics.csv`: aggregate MSE, MAE, R2 and pseudo-R2
- `tissue_metrics.csv`: tissue-level MSE, MAE and R2
- `predictions_long.csv`: true/predicted values by gene and tissue
- `plots/*.png`: aggregate and tissue-level prediction plots

**`module/`: directory for our customized modules**

- [`module/`](https://github.com/shanmukh0504/FloraBERT/tree/master/module/florabert): our main module named `florabert` that packages customized functions
  - [`config.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/config.py): project-wide configuration settings and absolute paths to important directories/files
  - [`dataio.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/dataio.py): utilities for performing I/O operations (reading and writing to/from files)
  - [`gene_db_io.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/gene_db_io.py): helper functions to download and process gene sequences
  - [`metrics.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/metrics.py): functions for evaluating models
  - [`nlp.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/nlp.py): custom classes and functions for working with text/sequences
  - [`training.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/training.py): helper functions that make it easier to train models in PyTorch and with Huggingface's Trainer API, as well as custom optimizers and schedulers
  - [`transformers.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/transformers.py): implementation of RoBERTa model with mean-pooling of final token embeddings, as well as functions for loading and working with Huggingface's transformers library
  - [`utils.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/utils.py): General-purpose functions and code
  - [`visualization.py`](https://github.com/shanmukh0504/FloraBERT/blob/master/module/florabert/visualization.py): helper functions to perform random k-mer flip during data processing and make model prediction

### Pretrained models

If you wish to experiment with our pre-trained FLORABERT models, you can find the saved PyTorch models and the Huggingface tokenizer files [here](https://drive.google.com/drive/folders/1qHwRfXxPVC1j2GcZ-wFOT3BmTmHRr_it?usp=sharing)

**Contents**:

- `byte-level-bpe-tokenizer`: Files expected by a Huggingface `transformers.PretrainedTokenizer`
  - `merges.txt`
  - `vocab.txt`
- transformer: Both language models can instantiate any RoBERTa model from Huggingface's `transformers` library. The prediction model should instantiate our custom `RobertaForSequenceClassificationMeanPool` model class
  1. `language-model`: Trained on all plant promoter sequences
  2. `language-model-finetuned`: Further trained on just maize promoter sequences
  3. `prediction-model`: Fine-tuned on the multitask regression problem

---

### Personal Updates on the Cloned Repository: (Removed data links due to low google drive space)

**First module has been completed. All data / outputs are under [`data`](https://github.com/shanmukh0504/FloraBERT/tree/main/data) or [`models`](https://github.com/shanmukh0504/FloraBERT/tree/main/models). Moving to Second Module. The following steps were essential for this [`script`](https://github.com/shanmukh0504/FloraBERT/tree/main/scripts).**

The following updates have been done using python scripts under [`3-RNAseq-quantification/`](https://github.com/shanmukh0504/FloraBERT/tree/main/scripts/3-RNAseq-quantification):

- The scripts under this module requires a lot of resources and time (patience). We opted to use the Bioinformatics website [`Galaxy`](https://usegalaxy.org). This provides every user 250GB storage and allows the ability to use a number of very useful and important bioinformatics tools.
- The scripts under the module dealt with 26 NAM lines / cultivars of Maize. We replicated the entire process under this module in this website, with some minor changes (not in output). The first step was to get all the runs corresponding to each cultivar and unique organsim part for each, to avoid repitition.
- This was achieved by getting the base data from [`EBI`](https://www.ebi.ac.uk/) and searching for the 2 Bioprojects mentioned in the supplementary material (under [`research_papers`](https://github.com/shanmukh0504/FloraBERT/tree/main/research_papers). This data was then used alongside the [`helper_codes`](https://github.com/shanmukh0504/FloraBERT/tree/main/helper_codes) scripts to get the file [`unique_orgs_runs.tsv`](https://github.com/shanmukh0504/FloraBERT/blob/main/helper_files/unique_orgs_runs.tsv). This file contains the runs corresponding to unique organism parts of each cultivar.
- A workflow was then created / implemented / configured (the base workflow was created by user vasquex11 on the mentioned website) to align with the scripts. The runs were first uploaded per cultivar to the website (after logging in) in txt format, one per line. Next, fasterq-dump tool was used with --split-files option selected to get the fastq files corresponding to the runs.
- The created workflow [`FloraBERT (Trimmomatic + HISAT2 + featureCounts)`](https://usegalaxy.org/workflows/run?id=2a7fbfc1c17e38bc) was used to perform all the actions mentioned in the module. The final output are the featureCounts files corresponding to each run ( extending to unique organsim part of cultivars ). The steps are self-explanatory (using the research papers).
