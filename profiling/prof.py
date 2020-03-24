import sys
sys.path.append("..")
import numpy as np
np.random.seed(0)
from SubCMedians.data_generator import make_subspace_blobs
from SubCMedians.subcmedians import subcmedians
D = 50
dataset_params={"p_dim": 0.7,
                "n_samples":5000,
                "n_features":D,
                "centers":12}
X,y_true,ss = make_subspace_blobs(**dataset_params)
X = (X - X.mean(axis=0))/ X.std(axis=0)

k = subcmedians(D, Gmax=300, H=200, nb_iter=50000)
k.fit(X)
