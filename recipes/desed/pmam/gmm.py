import argparse
import os
import pickle
from random import randint
from time import time
from typing import List

import numpy as np
import torch
from sklearn.decomposition import PCA
from pycave.clustering import KMeans
from pycave.bayes.gmm import GaussianMixture


def load_feature(data_path: List[str]) -> torch.Tensor:
    with open(data_path, "rb") as f:
        data_array = pickle.load(f)
        assert isinstance(data_array, torch.Tensor),\
            "type of data_arry is {}, but not toch.Tensor".format(type(data_array))
    return data_array


if __name__ == "__main__":
    # parser arguments
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--algorithm', type=str, default="GMM")
    parser.add_argument('--cluster_num', type=int)
    args = parser.parse_args()
    n_components = args.cluster_num
    dimension = args.dim  # dimension of the embedding
    data = load_feature(args.data_path)
    print("length of data is {}".format(data.shape[0]))
    data = data[torch.randperm(data.shape[0])]

    if data.shape[-1] > dimension:
        print("Execute the PCA algorithm to compresse the dimension from {0} to {1}".format(data.shape[-1], dimension))
        # Initialize PCA with the target number of dimensions
        pca = PCA(n_components=dimension)
        # Fit PCA on the data and transform it
        data = pca.fit_transform(data)
        print("Finish PCA algorithm")
    else:
        pca = None

    assert data.shape[-1] == dimension

    start_time = time()
    if args.algorithm == "GMM":
        model = GaussianMixture(
            num_components=n_components,
            covariance_type="full",
            batch_size=1500_000,
            trainer_params=dict(accelerator="gpu", devices=1),
        )
    elif args.algorithm == "kmeans":
        model = KMeans(
            num_clusters=n_components,
            batch_size=1000_000,
            trainer_params=dict(accelerator="gpu", devices=1),
        )
    else:
        raise RuntimeError("Unknown algorithm")
    model.fit(data)

    print("End Fitting after {0} min".format((time() - start_time) / 60))
    # save GMM model
    with open(os.path.join(args.model_dir, "gmm.pkl"), "wb") as f:
        pickle.dump(model, f)
    if pca is not None:
        with open(os.path.join(args.model_dir, "pca.pkl"), "wb") as f:
            pickle.dump(pca, f)
    with open(os.path.join(args.model_dir, "gmm_means.pt"), "wb") as f:
        if args.algorithm == "GMM":
            torch.save(model.model_.means, f)
        elif args.algorithm == "kmeans":
            torch.save(model.model_.centroids, f)
