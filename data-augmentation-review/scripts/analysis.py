from os.path import join
import numpy as np
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal, gaussian_kde
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

from mlresearch.utils import load_plt_sns_configs, generate_paths

_, _, ANALYSIS_PATH = generate_paths(__file__)
RND_SEED = 42


def gaussian_mesh(X, mesh):
    """
    Obtain the density estimation of a gaussian generative model within a mesh grid.

    Parameters:
        X: array-like, (n_samples, n_features)
        mesh: array-like, (m, m)
    """

    gaussian = multivariate_normal(mean=np.mean(X, axis=0), cov=np.cov(X.T))
    prob = gaussian.pdf(mesh)

    return prob


def gmm_mesh(X, mesh):
    """
    Obtain the density estimation of a gaussian mixture model within a mesh grid.

    Parameters:
        X: array-like, (n_samples, n_features)
        mesh: array-like, (m, m)
    """

    gmm = GaussianMixture(n_components=2, random_state=RND_SEED).fit(X)
    prob = gmm.score_samples(np.reshape(mesh, (100 * 100, 2)))
    prob = np.reshape(prob, (100, 100))

    prob = prob - prob.min()
    prob = prob / prob.sum()

    return prob


def kde_mesh(X, mesh):
    """
    Obtain the density estimation of a kernel density estimation model within a mesh
    grid.

    Parameters:
        X: array-like, (n_samples, n_features)
        mesh: array-like, (m, m)
    """
    kde = gaussian_kde(X.T)
    prob = kde.pdf(np.reshape(np.moveaxis(mesh, -1, 0), (2, 100 * 100)))
    prob = np.reshape(prob, (100, 100))

    return prob


if __name__ == "__main__":
    # Generate data and meshgrid
    X, y = make_blobs(
        n_samples=100,
        n_features=2,
        centers=2,
        cluster_std=[1, 2],
        random_state=RND_SEED,
    )

    x, y = [
        np.array(np.linspace(X.min(0)[i] - 1, X.max(0)[i] + 1, 100)) for i in range(2)
    ]
    mesh = np.stack(np.meshgrid(x, y), axis=2)

    load_plt_sns_configs(18)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))

    # Visualize original data
    sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=axes[0])

    # Visualize pdf methods
    for ax, generative_model in zip(axes[1:], [gaussian_mesh, gmm_mesh, kde_mesh]):
        prob = generative_model(X, mesh)
        ax.contourf(x, y, prob, alpha=0.5, cmap="Blues")
        sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax)
        ax.yaxis.set_ticklabels([])

    for ax, l in zip(axes, ["a", "b", "c", "d"]):
        ax.set_xlabel(f"({l})")

    fig.savefig(
        join(ANALYSIS_PATH, "pdf-example.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
