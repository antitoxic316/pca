import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

X = load_iris(as_frame=True).data
target = load_iris(as_frame=True).target

X = np.expand_dims(X.to_numpy(), -1)
target = np.expand_dims(target.to_numpy(), -1)

class PCA:
    def __init__(self):
        pass
    
    def standarize(self, X: np.array):
        return (X - X.mean())/X.std()

    def cov_matrix(self, X):
        return np.cov(pca.standarize(X).squeeze().T)

    def compute_feature_vector(self, X):
        covar = self.cov_matrix(self.standarize(X))
        eigenvalues, eigenvectors = np.linalg.eig(covar)
        
        indeces = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[indeces].T

        eig_vec1 = np.expand_dims(eigenvectors[0], -1)
        eig_vec2 = np.expand_dims(eigenvectors[1], -1)

        feature_vector = np.concatenate((eig_vec1, eig_vec2), axis = 1)
        return feature_vector

    def get_pca_coords(self, X):
        return np.matmul(X.squeeze(), self.compute_feature_vector(X)).T


pca = PCA()

coords = pca.get_pca_coords(X)

plt.scatter(coords[0], coords[1])
plt.show()