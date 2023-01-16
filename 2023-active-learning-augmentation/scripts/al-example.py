from os.path import join
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from research.utils import load_plt_sns_configs, generate_paths
from research.active_learning import UNCERTAINTY_FUNCTIONS
from research.data_augmentation import GeometricSMOTE, OverSamplingAugmentation

RND_SEED = 42
_, _, ANALYSIS_PATH = generate_paths(__file__)


def plot_decision_function(X, y, clf, ax, colors, title=None):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if not Z.all():
        colormap = ListedColormap([colors[0], colors[1]])
    else:
        colormap = ListedColormap([colors[1]])
    ax.contourf(xx, yy, Z, alpha=0.15, cmap=colormap)


def visualize_clf_dataset(
    X,
    y,
    sel_mask=None,
    colors=None,
    clf=None,
    X_res=None,
    y_res=None,
    ax=None,
    title=None,
):

    if ax is None:
        ax = plt.subplot()

    if clf is not None:
        plot_decision_function(X, y, clf, ax, colors)

    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="^", c="grey")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="x", c="grey")

    if sel_mask is not None:
        ax.scatter(
            X[sel_mask & (y == 0), 0],
            X[sel_mask & (y == 0), 1],
            marker="^",
            c=colors[0],
        )
        ax.scatter(
            X[sel_mask & (y == 1), 0],
            X[sel_mask & (y == 1), 1],
            marker="x",
            c=colors[1],
        )

    if (X_res is not None) and (y_res is not None):
        ax.scatter(
            X_res[y_res == 0, 0], X_res[y_res == 0, 1], marker="^", c="green", alpha=0.4
        )
        ax.scatter(
            X_res[y_res == 1, 0], X_res[y_res == 1, 1], marker="x", c="green", alpha=0.4
        )

    ax.set_title(title)

    return ax


def standard_al(X, y, axes, colors, n_samples_iter=5, random_state=None):

    rng = np.random.default_rng(random_state)

    # Initialization
    initial = rng.choice(range(len(y)), n_samples_iter)
    sel_mask = np.zeros(len(y)).astype(bool)
    sel_mask[initial] = True

    # Iterate
    for i in range(1, 4):
        clf = KNeighborsClassifier(n_neighbors=3).fit(X[sel_mask], y[sel_mask])

        visualize_clf_dataset(
            X, y, sel_mask=sel_mask, colors=colors, clf=clf, ax=axes[i - 1]
        )
        unc = np.zeros(y.shape)
        unc[~sel_mask] = UNCERTAINTY_FUNCTIONS["entropy"](
            clf.predict_proba(X[~sel_mask]) + 1e-99
        )
        sel_mask[np.argsort(unc)[-n_samples_iter:]] = True

    return axes


def augmentation_al(X, y, axes, colors, n_samples_iter=5, random_state=None):

    rng = np.random.default_rng(random_state)

    # Initialization
    initial = rng.choice(range(len(y)), n_samples_iter)
    sel_mask = np.zeros(len(y)).astype(bool)
    sel_mask[initial] = True

    # Iterate
    for i in range(1, 4):
        X_res, y_res = OverSamplingAugmentation(
            GeometricSMOTE(
                k_neighbors=5,
                truncation_factor=0.5,
                deformation_factor=0.5,
            ),
            value=4,
            random_state=random_state,
        ).fit_resample(X[sel_mask], y[sel_mask])
        clf = KNeighborsClassifier(n_neighbors=3).fit(X_res, y_res)

        visualize_clf_dataset(
            X,
            y,
            sel_mask=sel_mask,
            colors=colors,
            clf=clf,
            X_res=X_res[sel_mask.sum() :],
            y_res=y_res[sel_mask.sum() :],
            ax=axes[i - 1],
        )
        unc = np.zeros(y.shape)
        unc[~sel_mask] = UNCERTAINTY_FUNCTIONS["entropy"](
            clf.predict_proba(X[~sel_mask]) + 1e-99
        )
        sel_mask[np.argsort(unc)[-n_samples_iter:]] = True

    return axes


def generate_al_example_visualization():
    # General configs
    colors = {0: "red", 1: "blue"}
    load_plt_sns_configs(20)

    # Generate mock data
    X, y = make_classification(
        n_samples=170, n_features=2, n_redundant=0, class_sep=0.7, random_state=RND_SEED
    )

    # Make visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.tight_layout(h_pad=-0.5, w_pad=-1)
    # gs = GridSpec(2, 5, figure=fig)

    standard_al(X, y, axes[0], colors=colors, n_samples_iter=5, random_state=RND_SEED)
    augmentation_al(
        X, y, axes[1], colors=colors, n_samples_iter=5, random_state=RND_SEED
    )

    for i, ax in zip("abc", axes[1]):
        ax.set_xlabel(f"({i})")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # ax_data = fig.add_subplot(gs[:, :2])
    # visualize_clf_dataset(X, y, ax=ax_data, colors=colors)

    fig.savefig(
        join(ANALYSIS_PATH, "al-example.pdf"),
        format="pdf",
        bbox_inches="tight",
    )


generate_al_example_visualization()
