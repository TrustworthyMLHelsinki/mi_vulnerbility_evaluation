
import numpy as np
import os
import argparse
from typing import Union
import warnings
import pickle
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from cached_data_loader import CachedFeatureLoader
from lira import convert_logit_to_prob, calculate_statistic, log_loss
from utils import cross_entropy_loss 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
        parser = argparse.ArgumentParser()

        parser.add_argument("--results", help="Directory to load results from.")
        parser.add_argument("--dataset", help="Dataset to use.")
        parser.add_argument("--dataset_dir", help="Path to load the dataset", default=".")
        parser.add_argument("--examples_per_class", type=int, default=200, help="Test dataset size.")
        parser.add_argument("--feature_extractor", choices=["vit-b-16", "BiT-M-R50x1"], default="vit-b-16", help="Feature extractor to use.")
        parser.add_argument("--train_batch_size", "-b", type=int, default=128, help="Batch size.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.0025, help="Learning rate.")
        parser.add_argument("--epochs", "-e", type=int, default=40, help="Number of fine-tune epochs.")
        parser.add_argument("--test_batch_size", type=int, default=256, help="Batch size.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for datasets, trainloader and opacus")
        parser.add_argument("--num_models", type=int, default=1, help="Total number of models.")
        parser.add_argument("--train", type=bool, default=False, help="If True, train models for D_start_idx to D_stop_idx partitions.")
        parser.add_argument("--start_idx", type=int, default=0, help="The index of sample to start from.")
        parser.add_argument("--stop_idx", type=int, default=1, help="The index of sample to start from.")        
        args = parser.parse_args()

        ## ensure the directory to hold results exists
        results_directory = os.path.join(args.results, args.dataset, f"Seed={args.seed}", f"T={int(args.examples_per_class)}")
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        dataset_reader = CachedFeatureLoader(path_to_cache_dir=args.dataset_dir,
                                                    dataset=args.dataset,
                                                    feature_extractor = args.feature_extractor,
                                                    random_seed=args.seed
                                                    )
        if args.dataset == "cifar10":
            num_classes = 10

        feature_dim = dataset_reader.obtain_feature_dim()
        train_features, train_labels, class_mapping = dataset_reader.load_train_data(shots=args.examples_per_class, 
                                                                                         n_classes=num_classes)
                        
        print("Total Samples in training set: ", train_features.size(0)) 
        N = int(args.examples_per_class * num_classes)
        if not args.train:
            ## Build the dataset partitions for shadow models
            target_in_indices = np.zeros((args.num_models + 1, N), dtype=bool)
            for i in tqdm(range(args.num_models + 1)):
                selected_indices = np.random.binomial(1, 0.5, N).astype(bool)
                target_in_indices[i, selected_indices] = True 

            with open(os.path.join(results_directory,'in_indices_target.pkl'),"wb") as f:
                pickle.dump(target_in_indices, f)  
        else:
            with open(os.path.join(results_directory,'in_indices_target.pkl'),"rb") as f:
                target_in_indices = pickle.load(f)

            target_stats = np.zeros((args.stop_idx - args.start_idx, N))
            for i in tqdm(range(args.start_idx, args.stop_idx)):
                D_in = target_in_indices[i]
                x, y = train_features[D_in].to(DEVICE), train_labels[D_in].to(DEVICE)
                train_loader = DataLoader(
                                    TensorDataset(x, y),
                                    batch_size = args.train_batch_size,
                                    shuffle=True
                                ) 
                model = create_head(feature_dim=feature_dim, 
                                    num_classes=num_classes
                                    )
                _ = fine_tune_batch(model,
                                    train_loader,
                                    lr = args.learning_rate,
                                    epochs = args.epochs,
                                    )
                stats, _ = get_stat_and_loss_aug(model, train_features, train_labels.numpy())
                target_stats[i - args.start_idx, :] = stats.flatten()
            with open(os.path.join(results_directory,f'stats_target_m_in_{args.start_idx}_{args.stop_idx}.pkl'),"wb") as f:
                pickle.dump(target_stats, f)  
                
def prob_to_score(prob: Union[np.ndarray, float], eps=1e-12):
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, eps, 1.0 - eps)
    return np.log(prob / (1.0 - prob))


def create_head(feature_dim: int, num_classes: int):
    head = nn.Linear(feature_dim, num_classes)
    head.weight.data.fill_(0.0)
    head.bias.data.fill_(0.0)
    head.to(DEVICE)
    return head

def get_stat_and_loss_aug(
        model,
        x,
        y,
        sample_weight=None):
    """A helper function to get the statistics and losses.

    Here we get the statistics and losses for the images.

    Args:
        model: model to make prediction
        x: samples
        y: true labels of samples (integer valued)
        sample_weight: a vector of weights of shape (n_samples, ) that are
            assigned to individual samples. If not provided, then each sample is
            given unit weight. Only the LogisticRegressionAttacker and the
            RandomForestAttacker support sample weights.
        batch_size: the batch size for model.predict

    Returns:
        the statistics and cross-entropy losses
    """
    model.eval()
    losses, stat= [], []
    with torch.no_grad():
        logits = model(x).cpu().numpy()
    prob = convert_logit_to_prob(logits)
    losses.append(log_loss(y, prob, sample_weight=sample_weight))
    stat.append(calculate_statistic(prob, y, sample_weight=sample_weight, is_logits=False))
    return np.expand_dims(np.concatenate(stat), axis=1), np.expand_dims(np.concatenate(losses), axis=1)

def fine_tune_batch(model, train_loader, lr, epochs):
    print("Training the model.")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.type(torch.LongTensor).to(DEVICE)
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            logits = model(batch_images)
            loss = cross_entropy_loss(logits, batch_labels)
            loss.backward()     
            del logits
            optimizer.step()
            torch.cuda.empty_cache()
    return -1

if __name__ == "__main__":
    with warnings.catch_warnings():
        # PyTorch depreciation warning that is a known issue (see opacus github #328)
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook*"
        )

        main()
