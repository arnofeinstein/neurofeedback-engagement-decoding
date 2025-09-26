import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter
from typing import Callable, Iterable, Sequence, Tuple
from sklearn.base import clone

def _class_sample_weights(y: Sequence[int], mode: str | None = "balanced") -> np.ndarray:
    """
    Compute sample weights for classification tasks based on class frequencies.

    Parameters
    ----------
    y : Sequence[int]
        Sequence of class labels for each sample.
    mode : str or None, optional
        Weighting mode. If "balanced", weights are inversely proportional to class frequencies.
        If None, all samples are assigned equal weight. Default is "balanced".

    Returns
    -------
    np.ndarray
        Array of sample weights corresponding to each input label.

    Examples
    --------
    >>> y = [0, 0, 1, 1, 1]
    >>> _class_sample_weights(y)
    array([1.25, 1.25, 0.83333333, 0.83333333, 0.83333333])
    """
    if mode is None:
        return np.ones_like(y, dtype=float)
    counts = Counter(y)
    n = len(y)
    w = {cls: n / (len(counts) * counts[cls]) for cls in counts}
    return np.asarray([w[cls] for cls in y], dtype=float)

def elasticnet(

    files,
    build_matrix_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
    *,
    win_len: int = 200,
    step: int = 50,
    t_max: int | None = None,
    alpha: float = 1e-4,
    l1_ratio: float = 0.15,
    n_splits: int = 5,
    skip_start: int = 0,
    skip_end: int = 0,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 5,
    max_iter: int = 2000,
    tol: float = 1e-3,
    random_state: int | None = 42,
    verbose: int = 0,
    debug_shuffle_y: bool = False,
):
    """
    Perform sliding-window elastic net classification using SGD on time-series data.

    This function applies an elastic net-regularized logistic regression classifier
    (SGDClassifier) to data extracted from multiple files, using a sliding window
    approach. For each window, it builds a feature matrix and label vector using
    `build_matrix_fn`, standardizes features, and evaluates classification
    performance using stratified k-fold cross-validation. Optionally, labels can
    be shuffled for debugging.

    Parameters
    ----------
    files : list
        List of file paths or objects containing the data to be analyzed.
    build_matrix_fn : Callable[..., Tuple[np.ndarray, np.ndarray]]
        Function to build the feature matrix (X) and label vector (y) for a given
        time window. Must accept `files`, `t_start`, and `t_stop` as arguments.
    win_len : int, optional
        Length of each sliding window (in samples or time units), by default 200.
    step : int, optional
        Step size between consecutive windows, by default 50.
    t_max : int, required
        Maximum time (or sample index) to consider for windowing.
    alpha : float, optional
        Regularization strength for elastic net, by default 1e-4.
    l1_ratio : float, optional
        Ratio between L1 and L2 regularization, by default 0.15.
    n_splits : int, optional
        Number of cross-validation folds, by default 5.
    skip_start : int, optional
        Number of initial samples/time units to skip, by default 0.
    skip_end : int, optional
        Number of final samples/time units to skip, by default 0.
    early_stopping : bool, optional
        Whether to use early stopping during SGD training, by default True.
    validation_fraction : float, optional
        Fraction of training data for validation in early stopping, by default 0.1.
    n_iter_no_change : int, optional
        Number of iterations with no improvement to wait before stopping, by default 5.
    max_iter : int, optional
        Maximum number of iterations for SGD, by default 2000.
    tol : float, optional
        Tolerance for stopping criteria, by default 1e-3.
    random_state : int or None, optional
        Random seed for reproducibility, by default 42.
    verbose : int, optional
        Verbosity level for progress reporting, by default 0.
    debug_shuffle_y : bool, optional
        If True, shuffle labels for debugging, by default False.

    Returns
    -------
    windows : list of tuple
        List of (start, stop) tuples for each window.
    centers : np.ndarray
        Array of window center times/indices.
    aucs : np.ndarray
        Array of mean cross-validated AUC scores for each window.
    betas : np.ndarray
        Array of mean classifier coefficients (betas) for each window.

    Raises
    ------
    ValueError
        If `t_max` is not provided, if input shapes are incorrect, or if labels are not binary.
    """
    
    if t_max is None:
        raise ValueError("t_max must be provided")

    task_start = skip_start
    task_stop = t_max - skip_end
    windows = [(t0, t0 + win_len) for t0 in range(task_start, task_stop - win_len + 1, step)]

    centers, aucs, betas = [], [], []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)

    base_estimator = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
        learning_rate="optimal",
        average=True,
    )

    for (t0, t1) in tqdm(windows, desc="Sliding windows (EN + zscore, SGD)", disable=verbose <= 0):
        X, y = build_matrix_fn(files, t_start=t0, t_stop=t1)

        if debug_shuffle_y:
            y = rng.permutation(y)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(f"Expected (n_samples, n_features) and (n_samples,), got {X.shape} / {y.shape}")
        if np.unique(y).shape[0] != 2:
            raise ValueError(f"Labels must be binary, got {np.unique(y)}")

        fold_scores, fold_coefs = [], []

        for train_idx, test_idx in cv.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            est = clone(base_estimator)
            sample_weight = _class_sample_weights(y[train_idx], mode="balanced")
            est.fit(X_train, y[train_idx], sample_weight=sample_weight)

            y_prob = est.predict_proba(X_test)[:, 1]
            fold_scores.append(roc_auc_score(y[test_idx], y_prob))
            fold_coefs.append(est.coef_.ravel())

        centers.append(0.5 * (t0 + t1))
        aucs.append(float(np.mean(fold_scores)))
        betas.append(np.mean(fold_coefs, axis=0))

    return windows, np.asarray(centers), np.asarray(aucs), np.vstack(betas)
