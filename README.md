## Code for ``On Reliability of Membership Inference Vulnerability Evaluation``

### Dependencies:

Install dependencies with `pip install -r requirements.txt`. 

## Data:

- **Adult:** Download the dataset from https://archive.ics.uci.edu/dataset/2/adult. Run `python TABPFN/preprocess_adult.py` to combine the downloaded splits, remove rows with missing values, and write `data/adult/adult.csv` with 45,222 records.
- **Patch Camelyon/CIFAR10:** obtain the data and cached features for head-only fine-tuning setup using
  [this repository](https://github.com/DPBayes/impact-dataset-properties-MI-vulnerability-deep-TL).

## Plotting notebooks:

In the workspace's `PLOTS` directory:
- `plot_auc_ccdf.ipynb`: Figures 2;
- `plot_variance_fits.ipynb`: Figure 3;
- `plot_lira_pp_fpc.ipynb`: Figure 4.
