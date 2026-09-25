
import numpy as np
import os
import argparse
from typing import Union
import warnings
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tqdm import tqdm
from utils import load_dataset

def main():
        parser = argparse.ArgumentParser()

        parser.add_argument("--results", help="Directory to load results from.")
        parser.add_argument("--dataset", help="Dataset to use.", default="breast_cancer", choices=["breast_cancer", "blood", "creditg", "diabetes", "heart", "adult-balanced"])
        parser.add_argument("--dataset_dir", help="Path to load the dataset", default=".")
        parser.add_argument("--target_dataset_size", "-tb", type=int, default=200, help="Test dataset size.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--num_target_models", type=int, default=1, help="Total number of target models.")
        parser.add_argument("--train_shadow", dest="train_shadow", default=False, action="store_true", help="If True, compute shadow model scores for sample (X,y).")
        parser.add_argument("--num_shadow_models", type=int, default=64, help="Total number of [IN + OUT] shadow models/sample.")
        parser.add_argument("--start_idx", type=int, default=0, help="The index of sample to start from.")
        parser.add_argument("--stop_idx", type=int, default=1, help="The index of sample to start from.")        
        args = parser.parse_args()

        ## ensure the directory to hold results exists
        directory = os.path.join(args.results, args.dataset, f"Seed={args.seed}", f"T={int(args.target_dataset_size)}")
        if not os.path.exists(directory):
            os.makedirs(directory)

        if args.dataset == 'breast_cancer':
            X, y = load_breast_cancer(return_X_y=True)
        elif args.dataset == 'adult-balanced':
            X = np.load(os.path.join(args.dataset_dir,"adult_balanced_clean_X.npy"))
            y = np.load(os.path.join(args.dataset_dir,"adult_balanced_clean_y.npy"))
            feature_names = np.load(os.path.join(args.dataset_dir,"adult_balanced_clean_feature_names.npy"), allow_pickle=True)

        else:
            X, y = load_dataset(args.dataset, args.dataset_dir)

        X_target_set, X_shadow_set, y_target_set, y_shadow_set = train_test_split(X, y, train_size=args.target_dataset_size, random_state=args.seed)
        print("Using target dataset of size:",X_target_set.shape[0])
        N, _ = X_target_set.shape
        if not args.train_shadow:
            target_in_indices = np.zeros((args.num_target_models + 1, N), dtype=bool)
            target_stats = np.zeros((args.num_target_models + 1, N))
            for i in tqdm(range(args.num_target_models + 1)):
                selected_indices = np.random.binomial(1, 0.5, N).astype(bool)
                target_in_indices[i, selected_indices] = True
                model = TabPFNClassifier(device="cuda")
                model.fit(X_target_set[selected_indices, :], y_target_set[selected_indices])
                predicted_probs = model.predict_proba(X_target_set)
                predicted_probs = np.take_along_axis(predicted_probs, y_target_set.reshape((-1, 1)), axis=1)
                target_stats[i, :] = prob_to_score(predicted_probs.flatten())

            target_stats = target_stats.reshape((args.num_target_models + 1, N, 1))

            with open(os.path.join(directory,'in_indices_target.pkl'),"wb") as f:
                pickle.dump(target_in_indices, f)    

            with open(os.path.join(directory,'stats_target.pkl'),"wb") as f:
                pickle.dump(target_stats, f)  
        else:
            in_shadow_stats = np.zeros((args.num_shadow_models, args.stop_idx - args.start_idx)) 
            out_shadow_stats = np.zeros((args.num_shadow_models, args.stop_idx - args.start_idx))
            n_member = N // 2 # expected target dataset size
            for i in tqdm(range(args.start_idx, args.stop_idx)):
                in_stats_i, out_stats_i = get_shadow_stats(X_target_set[i,:], y_target_set[i], 
                                           X_shadow_set, y_shadow_set, args.num_shadow_models, n_member)
                in_shadow_stats[:, i - args.start_idx] = in_stats_i
                out_shadow_stats[:, i - args.start_idx] = out_stats_i
            
            in_shadow_stats = np.expand_dims(in_shadow_stats, axis=2)
            out_shadow_stats = np.expand_dims(out_shadow_stats, axis=2)

            with open(os.path.join(directory,f'stats_shadow_idxs_{args.start_idx}_{args.stop_idx}.pkl'),"wb") as f:
                pickle.dump({"in": in_shadow_stats, "out": out_shadow_stats}, f)              
                
def prob_to_score(prob: Union[np.ndarray,float]):
    return np.log(prob / (1 - prob))

def sample_subset(X: np.ndarray, y: np.ndarray, n: int, replace: bool=True) -> tuple[np.ndarray, np.ndarray]:
    inds = np.random.choice(X.shape[0], n, replace=replace)
    return X[inds, :], y[inds]

def get_shadow_stats(X_target: np.ndarray, y_target: np.ndarray, 
                     X_shadow_set: np.ndarray, y_shadow_set: np.ndarray, 
                     num_shadow_models: int, n_member: int) -> [np.ndarray,np.ndarray]:
    

    in_predicted_probabilities = np.zeros((num_shadow_models))
    out_predicted_probabilities = np.zeros((num_shadow_models))

    for i in range(num_shadow_models):
        X_shadow, y_shadow = sample_subset(X_shadow_set, y_shadow_set, n_member)
        out_model = TabPFNClassifier(device="cuda")
        out_model.fit(X_shadow, y_shadow)
        predicted_probs = out_model.predict_proba(X_target.reshape(1,-1))
        out_predicted_probabilities[i] = predicted_probs[0][y_target]

        X_shadow[0, :] = X_target
        y_shadow[0] = y_target
        in_model = TabPFNClassifier(device="cuda")
        in_model.fit(X_shadow, y_shadow)
        in_predicted_probabilities[i] = in_model.predict_proba(X_target.reshape(1,-1))[0][y_target]

    in_stats = prob_to_score(in_predicted_probabilities)
    out_stats = prob_to_score(out_predicted_probabilities)
    return in_stats, out_stats



if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )

        main()