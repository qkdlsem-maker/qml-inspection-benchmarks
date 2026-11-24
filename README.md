# Quantum and Classical Kernels for Defect Retrieval Under Daily Inspection Quotas

Research code and aggregate results for the manuscript by **Hyerim Choi** and **Tae-Kook Kim** (corresponding author), Pukyong National University.

The experiments rank manufactured units for inspection when a plant can inspect only a fixed number of units each day. They compare an ideal four-qubit state-vector kernel, its uncoupled variant, classical radial basis function (RBF) kernels, and process-aware controls. All quantum computations run in Python on a CPU.

## Run the main benchmark

The recorded notebook outputs used 64-bit Python 3.14.0 on Windows. From the repository root, create and activate a virtual environment with that Python version. On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` to activate it. On macOS or Linux, use `source .venv/bin/activate`.

```sh
python -m venv .venv
# Activate the environment as described above.
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open [Inspection_Budget_Benchmarks.ipynb](Inspection_Budget_Benchmarks.ipynb), restart the kernel, and run all cells from top to bottom. The notebook downloads missing inputs and checks their hashes. To use local copies, place the files listed below in `data/` at the repository root. Set `export_folder = Path("inspection_benchmark_results")` in the last cell to regenerate CSVs while keeping the archived results in `evidence/` for comparison.

**Check the primary result first.** For the two original ProBayes test dates, 20 and 16 inspections were available. Among 60 defective parts, the coupled four-qubit kernel found **2**, the matched four-component RBF found **24**, and the uncoupled four-qubit kernel found **15**. In [benchmark_table.csv](evidence/benchmark_table.csv), select `dataset=ProBayes`, `percent=10`, `policy=daily`. These are observed counts with chronological resolution of tied scores. The notebook also reports expected counts under random selection within exact ties.

## Reproduction check

A fresh IPython process executed all 17 code cells of the main notebook with the recorded Windows/Python package versions and locally cached, hash-checked inputs. Exporting the newly computed results reproduced all 19 archived main-benchmark CSVs byte for byte. Eight die-casting script invocations reproduced the 13 checked die-casting evidence files byte for byte. The five code cells in the SECOM development notebook reproduced their stored output text. Rendering Figure 1 from the regenerated values produced the exact notebook PNG at 400 dpi and manuscript PNG at 600 dpi. These checks used the analyzed CSVs, including the third-party ProBayes conversions.

## Find each manuscript result

| Item | Procedure and output |
|---|---|
| Table I. Training and evaluation cohorts | The main notebook prints the training/test sizes, positive counts and number of test dates after loading the three cohorts. |
| Table II. Detected defects at 10% capacity | Filter [benchmark_table.csv](evidence/benchmark_table.csv) and [extra_trees_counts.csv](evidence/extra_trees_counts.csv) to 10%. The manuscript gives pooled and daily counts, in that order. |
| Table III. Defective parts and production settings | [probayes_setting_coverage.csv](evidence/probayes_setting_coverage.csv) gives observed parts and settings reached, and the expected number of settings. Sum ProBayes daily `tie_expected` in [group_detail.csv](evidence/group_detail.csv) at 10% for expected part counts. |
| Table IV. Allocation sensitivity | Compare the Hamilton and running allocations in [operational_daily_quotas.csv](evidence/operational_daily_quotas.csv) and [operational_rankings.csv](evidence/operational_rankings.csv). |
| Table V. Die-casting detections | Compare the [first 20 dates](external/evidence/initial/external_summary.csv) and [next 19 dates](external/evidence/continuation/external_summary.csv) for the `Largest remainder` policy. Tree controls are in [casting_trees.csv](external/evidence/casting_trees.csv), under `All eligible shots`. |
| Table VI. Detections among unmarked shots | Use the recalculated inspection quotas for the unmarked subset in the [first window](external/evidence/trial_flag_sensitivity_primary.csv) and [second window](external/evidence/trial_flag_sensitivity_continuation.csv), with the matching unmarked-subset rows in [casting_trees.csv](external/evidence/casting_trees.csv). |
| Figure 1. Defects found as daily capacity increases | Run the main notebook section **Daily inspection across budgets**. Set `figure_path` there to save the figure. The stored notebook figure uses 400 dpi. The manuscript uses the same plotted values at 600 dpi. |

The [SECOM development notebook](development/SECOM_Baseline_Pilot.ipynb) documents how kernel settings were selected before the October test. Other main-notebook outputs include pooled average precision, numerical-precision sensitivity, bounds across ties, and the November transfer results. All 303 target values in the analyzed November file were zero, so that campaign cannot distinguish methods by defect retrieval.

## Run the die-casting experiments

In JupyterLab open [the initial evaluation](external/DieCasting_Independent_Validation.ipynb) and [the temporal follow-on](external/DieCasting_Follow_On.ipynb). The second notebook can reconstruct the initial fit on its own. To run the scripts instead, use the following sequence from the repository root. The four freeze/evaluate commands write working outputs in `external/diecasting_results/` and `external/diecasting_followon_results/`.

```sh
cd external
python external_diecasting.py freeze
python external_diecasting.py evaluate
python followon_diecasting.py freeze
python followon_diecasting.py evaluate
python casting_trees.py
```

The first run freezes out-of-sample scores before joining held-out outcomes. The follow-on verifies those scores, keeps the original fit, and evaluates strictly later dates from the same production stream. The coupled kernel detected **883** and **516** defects over these windows, compared with **689** and **432** for the matched RBF. The tree analysis was added after observing the main casting results. For the flag sensitivity, use [trial_reproduction.txt](external/trial_reproduction.txt) and supply the local source files and the score/selection files from the respective freezes. `python verify_source_prefix.py train.csv competition_full.csv` independently compares the overlapping source rows when both files are available locally.

## Obtain the input data

The repository distributes experiment code, fixed analysis specifications and aggregate measurements. The source providers host the raw data under their own terms. Download from them or use the built-in pinned downloads, and observe the attribution and access terms of each source.

| Input | Source and analyzed version |
|---|---|
| SECOM | McCann and Johnston, UCI, 2008. Download the two files `secom.data` and `secom_labels.data` from the [official dataset](https://doi.org/10.24432/C54305). |
| KAMP injection molding | Ministry of SMEs and Startups, Republic of Korea, 2020. See the [official catalog](https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=&DATASET_SEQ=4&EQUIP_SEL=&FILE_TYPE_SEL=&GUBUN_SEL=&WDATE_SEL=&page=1) and the [labeled CSV mirror](https://github.com/sanngrok/molding-defect-prediction). The loader reads `labeled_data.csv` or downloads Git blob `91c63810c331a88c73e4ecd491429f35e1635199`. |
| ProBayes, first campaign | Lambers, Schüder and Krauß, [official v2 record](https://doi.org/10.23728/b2share.4c5692b886db419180f716acf895bf06). The analyzed input was this [CSV conversion](https://github.com/ar363/plastic-injection-moulding-quality-detection/blob/8741433081eeff1d628d749861aedaacb2a6fa6e/data/dataset.csv). Save it as `probayes.csv`. |
| ProBayes, November 2022 campaign | Lambers and colleagues, [official v1 record](https://doi.org/10.23728/b2share.3f80952ce5ff4be88ae4cf6a3bdfe732). The default analysis uses this [separate CSV conversion](https://github.com/rajrejin/Injection-Molding-Causal-Inference/blob/115bf535e3ae84ab7374b330108419a5c6a9414b/Data/Dataset.csv), named `probayes_nov2022_mirror.csv` locally. The notebook can also read the original Parquet with the optional package `pyarrow==23.0.1`. Row-by-row agreement of the two campaigns' analyzed CSVs with their original Parquet files has not been established. |
| Metal die casting | The initial [competition-training file](https://github.com/yeonduhaeyo/diecasting-project/blob/3010621441321d2d51f8043d0edc4ac6440cd80f/data/train.csv) has 73,612 rows. The [fuller competition mirror](https://github.com/greatsangho/2024-Kamp-Competition/tree/ad2b1147ea73fcdd54fed0507ed2f73da3c652b9/data) has 92,015. Exact repository paths, commits, encodings, blob IDs and SHA-256 digests are in [source_prefix_summary.json](external/evidence/source_prefix_summary.json). |

The main notebook verifies four SHA-256 digests for the original three cohorts and a separate Git blob ID and SHA-256 for the November CSV. The die-casting code accepts `--data` and, for the follow-on, `--full-data` to specify local inputs. The notebooks expose equivalent settings in their first code cells.

## Analysis records

The [original ProBayes specification](evidence/probayes_original_protocol.json) documents the prescoring 20/16-slot contrast and what was already known about the cohorts. The [November protocol](evidence/future_protocol.json) and [source amendment](evidence/future_protocol_source_amendment.json) record the separate transfer check. The [die-casting specifications](external/protocol/) and [score-freeze records](external/evidence/) document the initial and follow-on windows. These are internal records. Their hashes check file integrity. They were not externally preregistered.
