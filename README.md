## Code for ``On Reliability of Membership Inference Vulnerability Evaluation``

### Dependencies:

Install dependencies with `pip install -r requirements.txt`. 

## Data:

- **Adult:** Download the dataset from https://archive.ics.uci.edu/dataset/2/adult. Run `python TABPFN/preprocess_adult.py` to combine the downloaded splits, remove rows with missing values, and write `data/adult/adult.csv` with 45,222 records.
- **Patch Camelyon/CIFAR10:** obtain the data and cached features for head-only fine-tuning setup using
  [this repository](https://github.com/DPBayes/impact-dataset-properties-MI-vulnerability-deep-TL).

## FPC experiments:

In the workspace's `TABPFN` and `ViT` directories.

After obtaining the cached features, run these files in `ViT` in order:
1. `prepare_fpc_subsets.py` — prepare the $N_+$ frames and fixed evaluation targets.
2. `train_fpc_linear_finite_frame_models.py` — train $M$ finite-frame models, once per ratio.
3. `train_fpc_linear_loo_models.py` — train the population-proxy $M$ IN/OUT models per target.

Invoke these using `python -m ViT.<module_name>`.

For TabPFN experiments, run these files in order:

1. `preprocess_adult.py` — create `data/adult/adult.csv`.
2. `prepare_fpc_subsets_tabpfn.py` — prepare the nested frames and fixed evaluation targets.
3. `train_fpc_tabpfn_finite_frame_models_gpu.py` — fit finite-frame models, once per ratio.
4. `train_fpc_tabpfn_loo_models_gpu.py` — fit the population-proxy IN/OUT models.

Invoke these using `python -m TABFN.<module_name>`.

**[NOTE]** TabPFN experiments expect a CUDA-capable PyTorch setup as by default the code uses `TabPFNClassifier(device="cuda")`.

## Post-Processing experiments:

- For Adult/TabPFN, run `prepare_adult.py` from `PP` directory.
- For CIFAR10/Head, use the cached features.
- For CIFAR10/FiLM, use [CIFAR10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) dataset.


Generate scores and membership labels using,
- `train.py` (Adult/TabPFN);
- `train_linear_layer.py` (CIFAR10/Head);
-  `ResNet/train_models.py` (CIFAR10/FiLM).

## Plotting notebooks:

In the workspace's `PLOTS` directory.
- `plot_auc_ccdf.ipynb`: Figures 2;
- `plot_variance_fits.ipynb`: Figure 3;
- `plot_lira_pp_fpc.ipynb`: Figure 4.
