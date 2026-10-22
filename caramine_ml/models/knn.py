import numpy as np

class KNN:
    """
    General-purpose K-Nearest Neighbors (KNN).

    Supports:
        - Classification
        - Regression
        - Euclidean, Manhattan, and Minkowski distances
        - Uniform and distance-based weighting
        - Single-sample and batch prediction
        - Classification accuracy and regression R² scoring

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors.

    task : str, default="classification"
        Either "classification" or "regression".

    metric : str, default="euclidean"
        Distance metric: "euclidean", "manhattan", or "minkowski".

    p : float, default=2
        Minkowski distance parameter. p=1 gives Manhattan distance;
        p=2 gives Euclidean distance.

    weights : str, default="uniform"
        "uniform" gives equal weight to every neighbor.
        "distance" gives greater weight to closer neighbors.
    """

    def __init__(
        self,
        n_neighbors,
        task="classification",
        metric="euclidean",
        p=2,
        weights="uniform",
    ):
        self._validate_parameters(
            n_neighbors, task, metric, p, weights
        )

        self.n_neighbors = n_neighbors
        self.task = task
        self.metric = metric
        self.p = p
        self.weights = weights

        self.X_train = None
        self.y_train = None
        self.n_features_in_ = None
        self.classes_ = None

    # Paramenter validation

    @staticmethod
    def _validate_parameters(
        n_neighbors, task, metrics, p, weights
    ):
        if (
            isinstance(n_neighbors, (bool, np.bool_))
            or not isinstance(n_neighbors, (int, np.integer))
            or n_neighbors <=0
        ):
            raise ValueError("n_neighbors must be a positive integer.")

        if task.lower() not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")

        if metric.lower() not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError(
                "metric must be 'euclidean', "
                "'manhattan', or 'minkowski'."
            )

        if (
            isinstance(p, (bool, np.bool_))
            or not isinstance(p, (int, float, np.integer, np.floating))
            or not np.isfinite(p)
            or p <= 0
        ):
            raise ValueError("p must be a finite number >= 1")

        if weights.lower() not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")

    # Validate feature arrays
    @staticmethod
    def _validate_X(X, name='X'):
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must contain numerical values."
            ) from exc

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")

        if x.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one sample.")

        if x.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature.")

        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name} cannot contain NaN or infinite values.")

        return X

    # Fit: Store training examples

    def fit(self, X, y):
        X = self._validate_X(X, "X")
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional target array.")

        if len(X) != len(y):
            raise ValueError("X and y contain the same number of samples.")

        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed the training sample count.")

        if self.task == "classification":
            if y.dtype.kind not in ('U', 'S', 'i', 'u', 'r'):
                raise ValueError("classification labels must be strings or numbers.")

            if y.dtype.kind == "f" and not np.all(np.isfinite(y)):
                raise ValueError("classification labels cannot contain NaN or infinity.")

            if y.dtype.kind == "b":
                raise ValueError("Boolean classification labels are not supported.")

            self.classes_ = np.unique(y)
        else:
            try:
                y = np.asarray(y, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Regression targets must be numerical.") from exc

            if not np.all(np.isfinite(y)):
                raise ValueError("Regression targets cannot contain NaN or infinity.")

            self.classes_ = None

        self.X_train = X.copy()
        self.y_train = y.copy()
        self.n_features_in_ = X.shape[1]

        return self

    # Calculate distances

    def _calculate_distances(self, x):
        differences = np.abs(self.X_train - x)

        if self.metric.lower() == "euclidean":
            return np.sqrt(np.sum(differences**2, axis=1))

        elif self.metric.lower() == "manhattan":
            return np.sum(differences, axis=1)

        elif self.metric.lower() == "minkowski":
            return np.sum(
                differences**self.p, axis=1
            )**(1.0/self.p)

        return RuntimeError("Unsupported distance metric.")

    # Find Nearest neighbors

    def kneighbors(self, X, return_distance=True):
        if self.X_train is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                
            )