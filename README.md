## Code for ``On Reliability of Efficient Membership Inference Vulnerability Evaluation``

### Dependencies:

Install the Python dependencies with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm tueplots jupyter torch tensorflow tabpfn
```

`train_eff_lira_models.py` uses `TabPFNClassifier(device="cuda")`, so the
TabPFN experiments expect a CUDA-capable PyTorch setup. To run on CPU, change
the `device="cuda"` argument in that file.

### Preparing Datasets:

Use `prepare_adult.py` to download, clean, balance, and encode the UCI Adult
dataset:

```bash
python prepare_adult.py --out-dir adult_balanced_npy --train-size 10000 --random-state 42
```

If the Adult raw files are already available locally, pass both raw file paths:

```bash
python prepare_adult.py \
  --raw-train-path path/to/adult.data \
  --raw-test-path path/to/adult.test \
  --out-dir adult_balanced_npy
```

The script writes `X.npy`, `y.npy`, train/test splits, feature metadata, and
`metadata.json` to the output directory.

Other datasets are loaded through `utils.py`. The supported dataset in
`train_eff_lira_models.py` include `blood`, `creditg`, `diabetes`, `heart`, and
`adult-balanced`. 

### Running Efficient LiRA:

Use `train_eff_lira_models.py` in two steps. First create the membership matrix using,

```bash
python train_eff_lira_models.py \
  --results results \
  --dataset adult-balanced \
  --dataset_dir adult_balanced_npy \
  --target_dataset_size 10000 \
  --seed 42 \
  --num_models 10000
```

Then train TabPFN models for an index range:

```bash
python train_eff_lira_models.py \
  --results results \
  --dataset adult-balanced \
  --dataset_dir adult_balanced_npy \
  --target_dataset_size 10000 \
  --seed 42 \
  --num_models 10000 \
  --train True \
  --start_idx 0 \
  --stop_idx 100
```

This writes files such as
`{results}/{dataset}/Seed={seed}/T={target_dataset_size}/stats_target_m_in_{start_index}_{stop_index}.pkl`. Run additional ranges if needed.

### Other Analysis Notebooks:

The notebooks have the following roles:

* `efficient_lira_with_pp.ipynb`: To compute efficient LiRA statistics with
  post-processing from saved `in_indices_target.pkl` and `stats_target.pkl`
  files.
* `fpc.ipynb`: For finite-population correction analysis.
* `plots.ipynb`: plotting code for saved experiment result files.

Some notebooks expect result files under `results/`. These result files are not
all included in the repository and should be generated with the scripts above or
provided separately.
